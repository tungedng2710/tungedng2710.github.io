# TONVERSE — Personal Site & Blog

Source for `tungedng2710.github.io`, built as a static Astro site.

## Local development

- Prerequisite: Node.js 22.12 or newer.
- Run `./buildlocalapp.sh`.
- Open <http://localhost:4321>.

You can also use `npm install`, followed by `npm run dev`.

## Production build

Run `npm run build`. The static site is written to `dist/`.

## Structure

- `src/content/blog/` — Markdown blog posts
- `src/pages/` — file-based pages and routes
- `src/components/` — shared UI
- `src/layouts/` — document layouts and metadata
- `src/styles/` — global Sass styles
- `public/assets/` — static images, PDFs, and downloadable code
- `src/data/projects.ts` — project cards

## GitHub Pages

Pushing `main` runs `.github/workflows/deploy.yml`. In the repository’s
**Settings → Pages**, set **Source** to **GitHub Actions**.

## Credits & License

- Original theme: [WhatATheme](https://thedevslot.github.io/WhatATheme/).
- License: see `LICENSE`.
