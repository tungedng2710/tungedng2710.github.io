# TONVERSE — Personal Site & Blog

Source code for [tungedng2710.github.io](https://tungedng2710.github.io), a personal site for writing about AI, mathematics, and software engineering.

## Tech stack

- [Astro 7](https://astro.build/) with TypeScript and static-site generation
- Markdown/MDX posts managed through Astro Content Collections
- Sass and Bulma for styling
- KaTeX with Remark/Rehype plugins for mathematical notation
- Mermaid for diagrams
- GitHub Actions and GitHub Pages for deployment

## Local development

Node.js 22.12 or newer is required.

```sh
npm install
npm run dev
```

Open <http://localhost:4321>.

## Production build

```sh
npm run build
```

The generated static site is written to `dist/`.

## Structure

- `src/content/blog/` — Markdown blog posts
- `src/pages/` — file-based pages and routes
- `src/components/` — shared UI
- `src/layouts/` — document layouts and metadata
- `src/styles/` — global Sass styles
- `public/assets/` — static images, PDFs, and downloadable code
- `src/data/projects.ts` — project cards

## GitHub Pages

Pushing to `main` runs `.github/workflows/deploy.yml` and deploys the static build to GitHub Pages.

## License

See [LICENSE](LICENSE).
