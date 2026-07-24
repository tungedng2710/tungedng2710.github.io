import type { CollectionEntry } from "astro:content";

export type BlogPost = CollectionEntry<"blog">;
export const POSTS_PER_PAGE = 12;

export function postSlug(post: BlogPost) {
  return post.id.replace(/^\d{4}-\d{2}-\d{2}-/, "");
}

export function postUrl(post: BlogPost) {
  return `/blog/${postSlug(post)}`;
}

export function formatDate(date: Date, locale = "en-US") {
  return new Intl.DateTimeFormat(locale, {
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(date);
}

export function primaryPosts(posts: BlogPost[]) {
  const primaryIds = new Map<string, string>();

  posts.forEach((post) => {
    const key = post.data.translationKey;
    if (!key) return;
    const current = primaryIds.get(key);
    if (!current || post.data.lang === "en") primaryIds.set(key, post.id);
  });

  return posts.filter(
    (post) => !post.data.translationKey || primaryIds.get(post.data.translationKey) === post.id,
  );
}

export function readTime(body = "") {
  const words = body.trim().split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.ceil(words / 180));
}

export function excerpt(body = "", maxLength = 300) {
  const plain = body
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[#>*_`~|\\[\]{}()-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  return plain.length > maxLength
    ? `${plain.slice(0, maxLength).trimEnd()}…`
    : plain;
}
