// Live Environment Intelligence Module

let envLocation = null;
let envWeatherInterval = null;
let envCache = null;

// Hybrid Location Management: Load -> Firebase -> LocalStorage
async function initEnvironmentIntelligence(uid) {
    console.log("[Env] Init Environment Intelligence");
    
    // Check Local Storage First for speed
    const local = localStorage.getItem('env_location');
    if (local) {
        envLocation = JSON.parse(local);
        renderWeatherLoading();
        fetchWeatherData();
    }
    
    // Async check Firebase
    if (uid && typeof window.db !== 'undefined' && typeof window.doc !== 'undefined') {
        try {
            const docRef = window.doc(window.db, "users", uid);
            const sn = await window.getDoc(docRef);
            if (sn.exists() && sn.data().location) {
                const fbLoc = sn.data().location;
                if (!local || local.lat !== fbLoc.lat || local.lon !== fbLoc.lon) {
                    envLocation = fbLoc;
                    localStorage.setItem('env_location', JSON.stringify(fbLoc));
                    fetchWeatherData(); // Refetch if updated
                }
            } else if (!local) {
                // First-time user: No local, no Firebase
                showLocationSetupModal();
            }
        } catch (e) {
            console.error("[Env] Firestore location error", e);
            if (!local) showLocationSetupModal();
        }
    } else if (!local) {
        showLocationSetupModal();
    }
    
    // Refresh every 10 min
    if(envWeatherInterval) clearInterval(envWeatherInterval);
    envWeatherInterval = setInterval(() => { if(envLocation) fetchWeatherData(true); }, 600000);
}

// ── UI Updates ──
function renderWeatherLoading() {
    document.getElementById('envTempSkeleton')?.classList.remove('hidden');
    document.getElementById('envTemp')?.classList.add('hidden');
    
    document.getElementById('envInsightSkeleton')?.classList.remove('hidden');
    document.getElementById('envInsightText')?.classList.add('hidden');
}

function renderWeatherUI(data) {
    if(data.error) {
        document.getElementById('envInsightText').innerHTML = `<span class="text-red-400">Live data unavailable.</span>`;
        return;
    }
    
    // Hydrate DOM
    const current = data.current;
    
    // Disable skeletons
    document.getElementById('envTempSkeleton')?.classList.add('hidden');
    document.getElementById('envTemp')?.classList.remove('hidden');
    document.getElementById('envInsightSkeleton')?.classList.add('hidden');
    document.getElementById('envInsightText')?.classList.remove('hidden');
    
    // Fill text
    document.getElementById('envTemp').innerHTML = `${Math.round(current.temperature_2m)}<span class="text-3xl text-white/50 tracking-normal ml-0.5 font-bold align-top">°C</span>`;
    document.getElementById('envHumidity').textContent = `${Math.round(current.relative_humidity_2m)}%`;
    document.getElementById('envWind').textContent = `${Math.round(current.wind_speed_10m)} km/h`;
    document.getElementById('envUV').textContent = `${Math.round(current.uv_index || 0)}`;
    document.getElementById('envRain').textContent = `${current.precipitation.toFixed(1)} mm`;
    
    // Dynamic Icon
    const wcode = current.weather_code || 0;
    let icon = "cloud"; // fallback
    if(wcode === 0) icon = "clear_day";
    else if(wcode <= 3) icon = "partly_cloudy_day";
    else if(wcode < 70) icon = "rainy";
    else if(wcode < 80) icon = "cloudy_snowing";
    else if(wcode > 80) icon = "thunderstorm";
    document.getElementById('envIcon').textContent = icon;
    
    // AI Insight
    if(data.plant_insight) {
        document.getElementById('envInsightText').textContent = data.plant_insight;
    }
}

// ── Data Fetch ──
async function fetchWeatherData(force = false) {
    if(!envLocation) return;
    
    const parsedName = window.innerWidth < 640 && envLocation.name.length > 15 
        ? envLocation.name.substring(0, 15) + "..."
        : envLocation.name;
    document.getElementById('envLocName').textContent = (parsedName || "Determining Location").toUpperCase();
    
    if(!force && envCache && (Date.now() - envCache.timestamp < 300000)) {
        renderWeatherUI(envCache.data);
        return;
    }
    
    try {
        const res = await fetch(`/api/weather/combined?lat=${envLocation.lat}&lon=${envLocation.lon}`);
        const data = await res.json();
        envCache = { timestamp: Date.now(), data: data };
        renderWeatherUI(data);
    } catch (e) {
        console.error("Weather fetch failed", e);
        document.getElementById('envInsightText').innerHTML = `<span class="text-orange-300">Live data unavailable gracefully. Using cached strategies.</span>`;
        if (document.getElementById('envTempSkeleton')) {
            document.getElementById('envTempSkeleton').classList.add('hidden');
        }
    }
}

// ── Modals & Interactions ──
function showLocationSetupModal() {
    const m = document.getElementById('modalEnvLocation');
    if(m) m.classList.remove('hidden');
}

function hideLocationSetupModal() {
    const m = document.getElementById('modalEnvLocation');
    if(m) m.classList.add('hidden');
}

async function useGPSLocation() {
    const btn = document.getElementById('btnGpsLoc');
    btn.innerHTML = `<span class="material-symbols-outlined animate-spin">refresh</span> Accessing...`;
    
    if ("geolocation" in navigator) {
        navigator.geolocation.getCurrentPosition(async (pos) => {
            const lat = pos.coords.latitude;
            const lon = pos.coords.longitude;
            try {
                // Reverse geocode via Open-Meteo or generic
                const geoRes = await fetch(`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=en`);
                const geoData = await geoRes.json();
                const name = geoData.city || geoData.locality || "Your Location";
                
                await saveLocation(lat, lon, name);
                hideLocationSetupModal();
                btn.innerHTML = `<span class="material-symbols-outlined">my_location</span> Use GPS Location`;
            } catch(e) {
                await saveLocation(lat, lon, "GPS Location");
                hideLocationSetupModal();
            }
        }, () => {
            alert("GPS access denied. Please use manual search.");
            btn.innerHTML = `<span class="material-symbols-outlined">my_location</span> Use GPS Location`;
        });
    }
}

async function useManualLocation() {
    const input = document.getElementById('inputManualLoc').value.trim();
    if(!input) return;
    
    // Grab the button without needing an explicit ID by querying onclick handler
    const btn = document.querySelector('button[onclick="useManualLocation()"]');
    if (btn) {
        btn.dataset.originalText = btn.innerHTML;
        btn.innerHTML = `<span class="material-symbols-outlined animate-spin text-[16px] align-middle mr-2 mt-[-2px]">progress_activity</span> Searching...`;
        btn.disabled = true;
        btn.classList.add('opacity-80', 'cursor-not-allowed');
    }
    
    try {
        const res = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(input)}&count=1`);
        const data = await res.json();
        
        if (btn) {
            btn.innerHTML = btn.dataset.originalText;
            btn.disabled = false;
            btn.classList.remove('opacity-80', 'cursor-not-allowed');
        }
        
        if(data.results && data.results.length > 0) {
            const match = data.results[0];
            const locName = match.name + (match.admin1 ? ", " + match.admin1 : ""); // Give better name length
            await saveLocation(match.latitude, match.longitude, locName);
            hideLocationSetupModal();
            document.getElementById('inputManualLoc').value = ''; // clean input
        } else {
            alert("Location not found.");
        }
    } catch(e) {
        if (btn) {
            btn.innerHTML = btn.dataset.originalText;
            btn.disabled = false;
            btn.classList.remove('opacity-80', 'cursor-not-allowed');
        }
        alert("Search failed.");
    }
}

async function saveLocation(lat, lon, name) {
    const newLoc = { lat, lon, name };
    envLocation = newLoc;
    localStorage.setItem('env_location', JSON.stringify(newLoc));
    
    if(window.firebaseUser && typeof window.db !== 'undefined' && typeof window.updateDoc !== 'undefined') {
        try {
            const docRef = window.doc(window.db, "users", window.firebaseUser.uid);
            await window.setDoc(docRef, { location: newLoc }, { merge: true });
        } catch(e) {
            console.warn("Could not sync location to Firestore", e);
        }
    }
    
    renderWeatherLoading();
    fetchWeatherData(true);
}

// ── Graph/Detailed UI ──
let envChartInstance = null;
function openDetailedWeather() {
    const m = document.getElementById('modalEnvDetails');
    if(m) m.classList.remove('hidden');
    
    if(!envCache || !envCache.data || !envCache.data.hourly) return;
    
    const hourly = envCache.data.hourly;
    // Next 24 items
    const times = hourly.time.slice(0, 24).map(t => new Date(t).getHours() + ":00");
    const temps = hourly.temperature_2m.slice(0, 24);
    const hums = hourly.relative_humidity_2m.slice(0, 24);
    
    if(!window.Chart) return;
    
    const ctx = document.getElementById('weatherChart').getContext('2d');
    if(envChartInstance) envChartInstance.destroy();
    
    envChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [
                {
                    label: 'Temperature (°C)',
                    data: temps,
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    yAxisID: 'y',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Humidity (%)',
                    data: hums,
                    borderColor: '#3b82f6',
                    yAxisID: 'y1',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: { type: 'linear', display: true, position: 'left' },
                y1: { type: 'linear', display: true, position: 'right', grid: {drawOnChartArea: false} }
            }
        }
    });
}
function closeDetailedWeather() {
    const m = document.getElementById('modalEnvDetails');
    if(m) m.classList.add('hidden');
}

// Expose globals
window.initEnvironmentIntelligence = initEnvironmentIntelligence;
window.showLocationSetupModal = showLocationSetupModal;
window.useGPSLocation = useGPSLocation;
window.useManualLocation = useManualLocation;
window.openDetailedWeather = openDetailedWeather;
window.closeDetailedWeather = closeDetailedWeather;
