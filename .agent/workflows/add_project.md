---
description: How to add a new project to the portfolio
---

# Add New Project Workflow

## 1. Project Classification
Determine the status:
*   **Done:** Use standard Purple styling.
*   **Active:** Use Green styling + Pulse effect.
*   **Planned/TFG:** Use Amber styling.

## 2. Update Timeline (`index.html`)
Location: `#experience` section.
*   Add a new node in the nested branch structure if it's a university project.
*   Else, add main node.
*   **Pattern:** `div.relative` > Dot > Year > Title > Description.

## 3. Create Project Card (`index.html`)
Location: `#projects` section.
*   **Structure:** Wrap the entire card content in an `<a>` tag pointing to the project page.
    *   `a.group.bg-slate-800...`
*   **Title:** Do NOT put an `<a>` inside the `<h3>`. Use `div.block.group` instead.
*   **Update Icon:** Use standard Lucide icons (`data-lucide`).
*   **Update Badge:**
    *   "Private Repo" (Amber)
    *   "Paused" (Slate)
    *   "Active" (Green)
*   **Update Tags:** Tech stack in `span.bg-purple-900/30`.

## 4. Create Detail Page (Optional)
If the project needs a full case study:
1.  Duplicate `mathsolver.html` or `kinova.html`.
2.  Rename to `project-name.html`.
3.  Update content (Hero, Architecture, Gallery).
4.  Link card in `index.html` to this file.

## 5. SEO & Deployment
1.  If new page created -> Add to `sitemap.xml`.
2.  `git add .`
3.  `git commit -m "feat: add project [name]"`
4.  `git push`
