/**
 * Florentix AI — Landing Page Scan Logic
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Handles the simple scan form on the landing page (index.html).
 * Uses the same /api/predict Vercel orchestrator as the dashboard.
 *
 * Includes client-side Canvas image compression (512×512, 80% JPEG)
 * to stay within Vercel's 4.5MB payload limit.
 */

// ─── Image Compression Engine ────────────────────────────────────────────────
async function compressImage(file, maxWidth = 512, maxHeight = 512, quality = 0.8) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = (event) => {
            const img = new Image();
            img.src = event.target.result;
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let width = img.width;
                let height = img.height;
                // Maintain aspect ratio while fitting within bounds
                if (width > height) {
                    if (width > maxWidth) { height = Math.round((height * maxWidth) / width); width = maxWidth; }
                } else {
                    if (height > maxHeight) { width = Math.round((width * maxHeight) / height); height = maxHeight; }
                }
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);
                canvas.toBlob((blob) => {
                    resolve(new File([blob], file.name.replace(/\.[^/.]+$/, "") + ".jpg", { type: "image/jpeg" }));
                }, "image/jpeg", quality);
            };
        };
    });
}

// ─── Main Prediction Function ────────────────────────────────────────────────
async function predictDisease() {
    const input = document.getElementById("imageInput");
    const file = input.files[0];

    if (!file) {
        alert("Please upload an image first.");
        return;
    }

    try {
        // Compress image before upload (prevents Vercel 4.5MB payload limit)
        const compressedFile = await compressImage(file);

        const formData = new FormData();
        formData.append("file", compressedFile);

        // Use relative path — works on both Vercel (production) and local dev
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.message || data.error);
        }

        document.getElementById("disease").innerText = data.prediction || data.condition || "Unknown";
        document.getElementById("confidence").innerText = data.confidence + "%";
        document.getElementById("description").innerText = data.analysis || data.description || "";

        const remediesList = document.getElementById("remedies");
        if (remediesList) {
            remediesList.innerHTML = "";
            (data.treatment || data.remedies || []).forEach(item => {
                const li = document.createElement("li");
                li.innerText = item;
                remediesList.appendChild(li);
            });
        }

        const careList = document.getElementById("careTips");
        if (careList) {
            careList.innerHTML = "";
            (data.prevention || data.care_tips || []).forEach(item => {
                const li = document.createElement("li");
                li.innerText = item;
                careList.appendChild(li);
            });
        }

        document.getElementById("result").classList.remove("hidden");

    } catch (error) {
        let errorMsg = "Error: Could not connect to AI engine.\n\n";

        if (error.message.includes("Failed to fetch")) {
            errorMsg += "❌ AI service is temporarily unavailable.\n";
            errorMsg += "Please try again in a few moments.";
        } else if (error.message.includes("Server error")) {
            errorMsg += "❌ Server returned an error.\n";
            errorMsg += "The AI is processing too many requests. Try again shortly.";
        } else {
            errorMsg += error.message || "Unknown error occurred";
        }

        alert(errorMsg);
        console.error("Detailed error:", error);
    }
}
