export const site = {
  title: "TONVERSE",
  description: "Craft the future",
  keywords: ["AI", "Mathematics"],
  author: {
    name: "Tung Nguyen",
    image: "/assets/images/shanghai1.JPG",
    url: "https://github.com/tungedng2710",
    about:
      "Researcher and engineer specializing in artificial intelligence, with a focus on generative AI and computer vision. I develop practical applications and conduct research at the intersection of deep learning and visual computing. Through this platform, I share technical insights, methodologies, and comprehensive analyses to advance understanding in the field of AI and machine learning.",
  },
  email: "tungnguyen99.tn@gmail.com",
  social: {
    linkedin: "https://www.linkedin.com/in/tungedng2710",
    github: "https://github.com/tungedng2710",
    youtube: "https://www.youtube.com/channel/UCfdJlJUx5UKzM9EYqh9faxQ",
  },
  analyticsId: "UA-163806439-1",
};

export function withBase(path: string) {
  const base = import.meta.env.BASE_URL;
  return `${base}${path.replace(/^\/+/, "")}`;
}
