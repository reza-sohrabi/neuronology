// Static build. The site URL and base path come from the environment so the same
// source builds for a GitHub Pages project site (default) or a custom domain later.
//   SITE_URL=https://example.org SITE_BASE=/ npm run build
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const site = process.env.SITE_URL ?? 'https://reza-sohrabi.github.io';
const base = process.env.SITE_BASE ?? '/neuronology/';

export default defineConfig({
  site,
  base,
  trailingSlash: 'always',
  output: 'static',
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
  },
  integrations: [mdx()],
});
