# Portfolio Layout Standards 📏

Use these snippets when creating new pages to ensure consistency across the site.

## 1. Navbar (Standard)
*For root-level pages (e.g., `projects.html`)*

```html
<nav class="fixed w-full z-50 bg-slate-900/80 backdrop-blur-md border-b border-slate-800">
    <div class="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <a href="index.html" class="font-bold text-xl font-mono text-purple-400">&lt;Hugo/&gt;</a>
        <div class="flex items-center gap-8 text-sm font-medium">
            <a href="index.html" class="hover:text-purple-400 transition-colors">Home</a>
            <a href="index.html#about" class="hover:text-purple-400 transition-colors">About</a>
            <a href="index.html#projects" class="hover:text-purple-400 transition-colors">Projects</a>
            <a href="log.html" class="hover:text-purple-400 transition-colors">DevLog</a>
            <a href="contact.html" class="hover:text-purple-400 transition-colors">Contact</a>
        </div>
    </div>
</nav>
```

**Note:** For the *active* page, replace the `<a>` tag with a `<span>` and add styling:
```html
<span class="text-purple-400 font-bold">CurrentPage</span>
```

---

## 2. Navbar (DevLog / Subpages)
*For `blog/` entries or deep pages. Note the branding change.*

```html
<nav class="fixed w-full z-50 bg-slate-900/80 backdrop-blur-md border-b border-slate-800">
    <div class="max-w-4xl mx-auto px-6 h-16 flex items-center justify-between">
        <a href="../log.html" class="font-mono text-lg font-bold text-slate-200 hover:text-purple-400 flex items-center gap-2 transition-colors">
            &lt;Hugo/Logs&gt;
        </a>
    </div>
</nav>
```

---

## 3. Footer (Global)
*Background color must be `#0b0f19` (matches body).*

```html
<footer class="py-8 text-center text-slate-500 text-sm bg-[#0b0f19] border-t border-slate-800 relative z-10">
    <a href="contact.html" class="hover:text-purple-400 transition-colors block mb-2">
        <p>&copy; 2025 Hugo Sevilla Martínez. Built with <i data-lucide="heart" class="w-3 h-3 inline text-red-500"></i> & Code.</p>
    </a>
    <a href="terminal.html" class="inline-flex items-center gap-1 text-slate-700 hover:text-green-500 transition-colors font-mono text-xs opacity-50 hover:opacity-100" title="System Terminal">
        <i data-lucide="terminal" class="w-3 h-3"></i> >_
    </a>
</footer>
```
