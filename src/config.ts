export const site = {
  title: "TONVERSE",
  description: "Craft the future",
  keywords: ["AI", "Mathematics"],
  author: {
    name: "Tung Nguyen",
    image: "/assets/images/shanghai1.JPG",
    url: "https://github.com/tungedng2710",
    about:
      "AI researcher and engineer focused on generative AI and computer vision. I build practical systems and share the methods, experiments, and ideas behind them.",
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
