import os
import shutil

base_path = r"d:\Antigravity Projects\cloned plant disease prediction ai\plant-disease-prediction-ai\frontend"
source_img = r"C:\Users\lenovo\.gemini\antigravity\brain\bc687331-9a4a-4622-b93d-57e8a7f3fb21\media__1775986819319.png" # The exact 2nd pic uploaded earlier, or the latest transparent one. Wait, let me use the standard previously uploaded flat graphic that the user loved, actually the user wanted the transparent one: florentix_favicon_geometric_1775986000477.png. Wait, I will use florentix_favicon_geometric_1775986000477.png because they specifically said "but in png, so that bg is transparent".

source_img = r"C:\Users\lenovo\.gemini\antigravity\brain\bc687331-9a4a-4622-b93d-57e8a7f3fb21\florentix_favicon_geometric_1775986000477.png"

# Write the image copies
shutil.copy(source_img, os.path.join(base_path, 'favicon.png'))
shutil.copy(source_img, os.path.join(base_path, 'logo.png'))

html_files = ["dashboard.html", "index.html", "login.html", "signup.html"]

for h in html_files:
    p = os.path.join(base_path, h)
    if os.path.exists(p):
        content = open(p, 'r', encoding='utf-8').read()
        content = content.replace("favicon.svg", "favicon.png")
        content = content.replace("logo.svg", "logo.png")
        if h == "dashboard.html":
            # Fix Sidebar overlay: The problem is `.premium-toggle` position or `absolute -right-12`
            content = content.replace('<button id="sidebarToggleBtn" onclick="toggleDoctorSidebar()" class="premium-toggle">', '<button id="sidebarToggleBtn" onclick="toggleDoctorSidebar()" class="premium-toggle" style="right: auto; left: 10px; top: 10px; position: fixed;">')
            
            # Fix Auth blank page issue.
            auth_block_old = """onAuthStateChanged(auth, async user => {
  if (!user) {
      window.firebaseUser = null;
      return (window.location.href = 'login.html');
  }
  window.firebaseUser = user;
  const name = user.displayName || user.email.split('@')[0];
  UI('userGreeting').innerHTML = `Hello, <span style="color:#6ee7b7">${name}</span>`;
  UI('userAvatar').textContent = name.substring(0,2).toUpperCase();
  UI('userNameChip').textContent = name;
  await loadScans(user.uid);
  if(typeof initEnvironmentIntelligence === 'function') initEnvironmentIntelligence(user.uid);
  
  // Restore SPA Section on REFRESH only (sessionStorage survives refresh, not logout/login)
  const hash = window.location.hash;
  if (hash === '#overview') {"""
            
            auth_block_new = """onAuthStateChanged(auth, async user => {
  if (!user) {
      window.firebaseUser = null;
      return (window.location.href = 'login.html');
  }
  window.firebaseUser = user;
  const name = user.displayName || user.email.split('@')[0];
  UI('userGreeting').innerHTML = `Hello, <span style="color:#6ee7b7">${name}</span>`;
  UI('userAvatar').textContent = name.substring(0,2).toUpperCase();
  UI('userNameChip').textContent = name;
  
  const finishInit = async () => {
      try { await loadScans(user.uid); } catch(e){}
      if(typeof initEnvironmentIntelligence === 'function') initEnvironmentIntelligence(user.uid);
      
      const hash = window.location.hash;
      if (hash === '#overview') {"""
            
            content = content.replace(auth_block_old, auth_block_new)
            
            auth_end_old = """      } else if (savedSection && typeof window.showSection === 'function') {
          window.showSection(savedSection);
      } else if(typeof window.showSection === 'function') {
          window.showSection('dashboard');
      }
  }
});"""
            
            auth_end_new = """      } else if (savedSection && typeof window.showSection === 'function') {
          window.showSection(savedSection);
      } else if(typeof window.showSection === 'function') {
          window.showSection('dashboard');
      }
  };
  
  if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', finishInit);
  } else {
      finishInit();
  }
});"""
            content = content.replace(auth_end_old, auth_end_new)
            
            # The sidebar button
            content = content.replace('class="premium-toggle"', 'class="premium-toggle fixed z-50 left-0 top-[200px]"')

        open(p, 'w', encoding='utf-8').write(content)
        
print("Replacements complete.")
