import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const blog = defineCollection({
  loader: glob({ base: "./src/content/blog", pattern: "**/*.{md,mdx}" }),
  schema: z.object({
    title: z.string(),
    pubDate: z.coerce.date(),
    image: z.string(),
    description: z.string(),
    tags: z.array(z.string()).default([]),
    authorName: z.string().default("Tung Nguyen"),
    authorUrl: z.url().default("https://github.com/tungedng2710"),
    lang: z.string().default("en"),
  }),
});

export const collections = { blog };
