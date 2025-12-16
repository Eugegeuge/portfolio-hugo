# 🗺️ Portfolio Roadmap (2025+)

Living document tracking the evolution of `eugegeuge.com`.

## ✅ Completed (Phase 1: Foundation)
*   [x] **Cyberpunk Design System:** Dark mode, neon gradients, glassmorphism.
*   [x] **Core Pages:** Home (`index.html`), Projects (`kinova.html`, `mathsolver.html`), Contact.
*   [x] **Interactive Terminal:** `terminal.html` with file system and help commands.
*   [x] **Technical SEO:** Canonical tags, JSON-LD (Person), Sitemap, Robots.txt.
*   [x] **Analytics:** GA4 Integration.
*   [x] **Custom 404:** Glitch effect with hidden easter egg.
*   [x] **Repo Refactor:** Professionalizing "Virtual-Kinova-Interface".

---

## 🚀 Phase 2: Interaction & Polish (Current Focus)

### 1. 3D Viewer Integration 🧊
**Goal:** Showcase CAD skills directly in the browser.
*   **Target:** `kinova.html`.
*   **Tech:** `Three.js` (for control) or `<model-viewer>` (Google, simpler).
*   **Asset:** Kinova Gen2 `.obj` / `.glb` model.



### 3. Internationalization (i18n) 🌍
**Goal:** Structure the site to support Spanish natively.
*   **Current:** Mixed/English.
*   **Plan:** Button `[ES] / [EN]` that swaps text content (JSON based or separate HTMLs).

---

## 🔮 Phase 3: Content & Automation (Future)

### 4. DevLogs / Engineering Notebook 📓
**Goal:** Technical blog to explain the "How", not just the "What".
*   *Article Ideas:* "Handling Singularities in VR Teleoperation", "Optimizing CNNs for Handwriting".

### 5. Live Status API 🟢
**Goal:** Real-time engineering status.
*   **Integration:** GitHub API or WakaTime.
*   **Display:** "Currently coding in: Python" or "Last commit: 2h ago".

### 6. TFG Drone Integration 🚁
**Goal:** Prepare portfolio section for the 2026 Drone Tracking project.
*   **Mockup:** Placeholder section with "Coming Soon" and theoretical architecture.

---

## 🧪 Phase 4: Experimental / Cool "Nice-to-Haves" (Brainstorming)

### 7. Konami Code Easter Egg 🎮
**Goal:** A secret for the true nerds.
*   **Action:** User types `↑ ↑ ↓ ↓ ← → ← → B A`.
*   **Effect:** Confetti, "God Mode" (infinite glitch?), or a hidden retro game overlay.

### 8. 3D Tilt Cards (Apple TV Style) 🧊
**Goal:** Premium feel for project cards.
*   **Tech:** Vanilla JS mouse tracking.
*   **Effect:** Cards tilt slightly towards the cursor, showing a dynamic sheen/reflection.

### 9. "Now Playing" Widget 🎵
**Goal:** Personality injection.
*   **Integration:** Last.fm API or Spotify.
*   **Display:** Small pill in the footer: "Coding to: [Song Name] - [Artist]".

### 10. Fake "System Status" Footer 📟
**Goal:** Sell the "Terminal/Cyberpunk" fantasy.
*   **Display:** Small numbers in the footer changing randomly.
*   **Metrics:** "CPU Temp: 45°C", "Memory: 12GB", "Uptime: 420h".

### 11. PDF Resume Generator 📄
**Goal:** Utility for recruiters.
*   **Action:** Button "Print CV" in `contact.html`.
*   **Tech:** `window.print()` with a specific `@media print` stylesheet that strips the black background and formats it as a clean A4 paper.

---
*Maintained by Agent Antigravity*
