# Future Features Brainstorm & Roadmap 🧠🚀

Documento para la evolución del portfolio `eugegeuge.com`.
Estado: **Fase 2 (Post-Lanzamiento)**

> ✅ **Implementado:**
> *   Terminal Linux Interactivo (`terminal.html`).
> *   Analytics y SEO Técnico.
> *   Estética Cyberpunk v1.0.

## 1. Visor 3D (Three.js/Spline) 🧊
**Concepto:** Integrar modelos robóticos reales en la web.
*   **Idea:** Un modelo 3D del brazo **Kinova** o el **Dron** en sus respectivas páginas.
*   **Interacción:** El usuario puede rotarlo y hacer zoom.
*   **Tech:** `model-viewer` (Google) para simplicidad o `Three.js` para control total.
*   **Valor:** Muestra habilidades de CAD y visualización web.

## 2. DevLogs / Cuaderno de Ingeniería 📓
**Concepto:** Profundizar más allá del "qué" hice, al "cómo" lo resolví.
*   **Contenido:** Artículos técnicos cortos (Snippets, diagramas de arquitectura).
*   **Ejemplos:**
    *   *"Cómo evité las singularidades en el control del Kinova VR".*
    *   *"Mi configuración de ROS 2 para baja latencia".*
*   **Formato:** Markdown renderizados dinámicamente o una página simple `/blog`.

## 3. Modo "Corporativo" (Toggle Theme) 👔
**Concepto:** Un botón de pánico para reclutadores conservadores.
*   **Acción:** Un interruptor (quizás en la Terminal: `sudo mode --corp`).
*   **Resultado:**
    *   Fondo blanco/gris limpio.
    *   Fuente Sans-serif estándar (Inter).
    *   Sin efectos de neón/glow.
    *   Convierte la web en un CV digital sobrio parecido a un PDF.

## 4. Skills Graph Interactivo (D3.js) 🕸️
**Concepto:** Evolucionar la "Grid" estática de skills a un grafo de conocimiento.
*   **Visual:** Nodos flotantes conectados por física.
*   **Lógica:** Al pasar el ratón por **Python**, se iluminan **OpenCV** y **ROS 2** (mostramos relaciones y dependencias).
*   **Tech:** D3.js o Vis.js.

## 5. Easter Eggs (Detalles Geek) 🥚
**Concepto:** Premiar la curiosidad.
*   **Konami Code:** (`↑↑↓↓←→←→BA`) desencadena una lluvia Matrix o un minijuego simple.
*   **Command Not Found:** Mensajes de error graciosos en la terminal.
*   **404 Page:** Una página de error personalizada ("Robot not found" o "Connection Lost").

## 6. API de Estado en Tiempo Real 🟢
**Concepto:** Widget "System Status" real.
*   **Integración:** Conectar con la API de GitHub o WakaTime.
*   **Display:** Mostrar "Last commit: 2h ago" o "Currently coding in: Python" en el footer o en la tarjeta del Hero.

---
*Last Updated: 2025-12-12*
