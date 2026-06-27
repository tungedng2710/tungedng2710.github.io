import { getCollection } from "astro:content";
import type { APIRoute } from "astro";
import { excerpt, formatDate, postUrl, readTime } from "../utils/posts";

export const GET: APIRoute = async ({ site }) => {
  const posts = (await getCollection("blog")).sort(
    (a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf(),
  );

  const data = posts.map((post) => ({
    title: post.data.title,
    tags: post.data.tags.join(", "),
    url: new URL(postUrl(post), site).toString(),
    date: formatDate(post.data.pubDate),
    description: post.data.description,
    content: excerpt(post.body),
    image: post.data.image,
    readtime: readTime(post.body),
  }));

  return new Response(JSON.stringify(data), {
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
};
