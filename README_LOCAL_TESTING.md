# Sindi's Data Lab – original Satellite layout

This version is built from your **original uploaded Jekyll website**, with the original `_includes`, `_layouts/page.html`, and `_sass` structure retained. It restores the missing CSS, JavaScript, and image assets from local source components so you can test locally. It is not the unrelated redesign.

## What changed

- Added a real root `index.html` (fixes WEBrick directory listing).
- Rebuilt `assets/css/style.scss` using the original Satellite SCSS partials and supplied replacement JS assets because the uploaded ZIP omitted the original `assets/css` and `assets/js` files.
- Kept the original sidebar, starry background, light/dark switch, search, category tree, post layout, and TOC. The restored client scripts implement core behavior rather than exactly reproducing upstream JavaScript.
- Added categories **Statistics**, **Guides**, and **Projects**, with one completed CS533 article and three clearly identified placeholder posts.
- Added the anime avatar in the original circular profile-image location.
- Kept notebook downloads **inside the article only**, under Further reading. There is **no notebook link in the sidebar or homepage**.
- Corrected placeholder analytics/comment counters and removed broken references to assets absent from the uploaded ZIP.
- Left the original theme attribution and license.

## Run it on Ubuntu

```bash
bundle install
bash scripts/check-site.sh
bundle exec jekyll serve --livereload
```

Open http://localhost:4000. Test homepage, /Statistics/, /Guides/, /Projects/, and the actual article, plus the theme switch, search, and in-article notebook link.

Use `git diff` to review before pushing. Avoid copying `_site/` or `.jekyll-cache/` to GitHub.
