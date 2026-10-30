export const experience = [
  {
    role: "AI Engineer",
    organization: "Viettel Solutions Corporation",
    logo: "/assets/images/companies/viettel-solutions.png",
    location: "Hanoi, Vietnam",
    period: "March 2022 — Present",
    type: "Full-time",
    highlights: [
      "Conduct research and development in intelligent document processing, with an emphasis on structured information extraction from forms, tables, scanned documents, and semi-structured layouts.",
      "Fine-tune vision-language models for document understanding and information extraction, addressing layout interpretation, key-field recognition, and structured output generation.",
      "Develop document layout analysis and table extraction modules to preserve reading order and support optical character recognition (OCR), information extraction, and validation workflows.",
      "Design and implement end-to-end license plate recognition pipelines encompassing image preprocessing, vehicle and plate detection, OCR, post-processing, and validation.",
      "Deploy license plate recognition models using NVIDIA Triton Inference Server and optimize inference workflows for traffic monitoring applications.",
      "Develop methods for traffic violation recognition through the integration of object detection, tracking, license plate recognition, and rule-based video analysis.",
      "Investigate and develop speech enhancement models for integration into the Viettel Meeting platform.",
    ],
  },
  {
    role: "Research Consultant",
    organization: "WorldQuant BRAIN",
    logo: "/assets/images/companies/worldquant-brain.png",
    location: "Remote",
    period: "March 2025 — Present",
    type: "Part-time",
    highlights: [
      "Formulate data-driven investment hypotheses and develop quantitative alpha signals, evaluating their behaviour through simulation on the WorldQuant BRAIN platform.",
    ],
  },
  {
    role: "AI Engineer Intern",
    organization: "Grooo International JSC",
    logo: "/assets/images/companies/grooo.png",
    location: "Hanoi, Vietnam",
    period: "October 2021 — March 2022",
    type: "Part-time",
    highlights: [
      "Investigated and evaluated face detection, alignment, and recognition methods for real-time computer vision applications.",
      "Examined object tracking algorithms under challenging visual conditions, including occlusion, motion blur, and crowded scenes.",
      "Prepared datasets, conducted model experiments, and contributed to the integration of computer vision modules into a camera monitoring system.",
      "Conducted performance evaluation and error analysis to identify opportunities for improving model reliability, inference efficiency, and robustness.",
    ],
  },
  {
    role: "Data Scientist Intern",
    organization: "PIXTA Vietnam Co., Ltd.",
    logo: "/assets/images/companies/pixta.png",
    location: "Hanoi, Vietnam",
    period: "December 2020 — June 2021",
    type: "Part-time",
    highlights: [
      "Investigated facial expression recognition and multi-task learning approaches for the joint prediction of facial attributes.",
      "Designed and conducted face recognition experiments encompassing dataset preparation, model training, evaluation, and comparative performance analysis.",
      "Annotated, curated, and organized image datasets to support the quality and consistency of training and validation data.",
      "Analyzed and visualized image datasets, label distributions, and model predictions to support error analysis and guide model refinement.",
    ],
  },
  {
    role: "AI Engineer Intern",
    organization: "Techainer JSC",
    logo: "/assets/images/companies/techainer.png",
    location: "Hanoi, Vietnam",
    period: "August 2020 — October 2020",
    type: "Part-time",
    highlights: [
      "Investigated and implemented CenterNet-based object detection for Vietnamese identity card localization in electronic Know Your Customer (eKYC) applications.",
      "Developed identity card alignment workflows incorporating cropping, rotation, and image normalization prior to text recognition.",
      "Applied optical character recognition (OCR) to extract identity information from Vietnamese identity cards.",
      "Evaluated the detection, alignment, and OCR pipeline under variations in illumination, capture angle, and background conditions.",
    ],
  },
];

export const publications = [
  {
    title: "VPHQA: Vietnamese Pregnancy Health Question Answering Dataset",
    venue: "EAI INISCOM, 2025",
    href: "https://doi.org/10.1007/978-3-032-02362-9_13",
    label: "DOI",
  },
  {
    title: "Chatbot and Reproductive Health for Minority Women",
    venue: "AI4SDF, 2025",
  },
  {
    title: "The Solution of Computer Vision for Combating COVID-19",
    venue:
      "Computational Intelligence Techniques for Green Smart Cities, Springer, 2021",
    href: "https://doi.org/10.1007/978-3-030-96429-0_9",
    label: "DOI",
  },
];

export const projects = [
  {
    title: "dMOM Chatbot",
    description:
      "A medical chatbot for the PHAD-sponsored dMOM project under Vietnam’s Ministry of Health. Fine-tuned large language models for Vietnamese medical question answering, built a retrieval-augmented generation pipeline grounded in trusted medical knowledge, and implemented a web application for interaction, demonstration, and evaluation.",
  },
  {
    title: "AI Traffic Analysis",
    description:
      "An end-to-end traffic-video analysis system combining vehicle detection and tracking, license-plate detection, and OCR. Trained and evaluated models on traffic datasets, integrated the modules into a unified backend pipeline, and built a web demo to visualize results.",
    href: "https://github.com/tungedng2710/AI-Traffic-Analysis",
  },
  {
    title: "Vietnamese ID Card Recognition",
    description:
      "An eKYC document-recognition pipeline for Vietnamese identity-card localization and alignment, with document cropping, perspective correction, image enhancement, text detection, OCR, and evaluation across varied capture conditions.",
  },
];

export const education = [
  {
    period: "2023 — 2025",
    degree: "Master of Science",
    field: "Applied Mathematics and Informatics",
    institution: "Hanoi University of Science and Technology",
    logo: "/assets/images/education/hust.png",
    result: "Excellent",
    cpa: "3.81",
    maximum: "4.00",
  },
  {
    period: "2017 — 2022",
    degree: "Engineer’s Degree",
    field: "Applied Mathematics and Informatics",
    institution: "Hanoi University of Science and Technology",
    logo: "/assets/images/education/hust.png",
    result: "Very Good",
    cpa: "3.46",
    maximum: "4.00",
  },
  {
    period: "2014 — 2017",
    degree: "High School Diploma",
    field: "",
    institution: "Chu Van An National High School",
    logo: "/assets/images/education/chu-van-an.jpg",
    result: "",
    cpa: "",
    maximum: "",
  },
];

export const toolkits = [
  {
    title: "Research areas",
    category: "Research",
    icon: "spark",
    items: [
      "Computer Vision",
      "Intelligent Document Processing",
      "Generative AI",
      "Reinforcement Learning",
    ],
  },
  {
    title: "Programming",
    category: "Engineering",
    icon: "code",
    items: ["Python", "C#", "C/C++", "MATLAB"],
  },
  {
    title: "AI / ML frameworks",
    category: "Engineering",
    icon: "layers",
    items: ["PyTorch", "TensorFlow", "MLflow", "vLLM"],
  },
  {
    title: "Computer vision",
    category: "Computer Vision",
    icon: "grid",
    items: [
      "Object Detection",
      "OCR",
      "Image Classification",
      "Tracking",
      "Segmentation",
      "Layout Analysis",
    ],
  },
  {
    title: "Generative AI",
    category: "Generative AI",
    icon: "spark",
    items: [
      "Large Language Models",
      "Vision-Language Models",
      "RAG",
      "Prompt Engineering",
      "Model Fine-tuning",
    ],
  },
  {
    title: "MLOps & tools",
    category: "Engineering",
    icon: "code",
    items: [
      "Docker",
      "Git",
      "Triton Inference Server",
      "Label Studio",
      "Roboflow",
      "Dify",
      "Ollama",
      "n8n",
    ],
  },
  {
    title: "Professional skills",
    category: "Research",
    icon: "globe",
    items: [
      "Research",
      "Technical Presentation",
      "Teamwork",
      "Leadership",
      "Scrum",
    ],
  },
];

export const credentials = [
  {
    title: "Quant Finance Explorer Program ’24",
    issuer: "WorldQuant",
    date: "Nov 2024",
  },
  {
    title: "Google Advanced Data Analytics",
    issuer: "Google",
    date: "Aug 2024",
  },
  { title: "Google Project Management", issuer: "Google", date: "Aug 2024" },
  { title: "Google AI Essentials", issuer: "Google", date: "Aug 2024" },
  {
    title: "IELTS Academic",
    issuer: "British Council · Overall 6.0",
    date: "May 2023",
  },
];

export const service = [
  {
    period: "2026",
    role: "Mentor",
    organization: "Viettel Digital Talent Program",
  },
  {
    period: "2023 — 2025",
    role: "Research Mentor",
    organization: "Scientific research competitions for high-school students",
  },
  {
    period: "Jan 2018 — Aug 2018",
    role: "Extra-class Assistant",
    organization:
      "SAMI study support club, Hanoi University of Science and Technology",
  },
  {
    period: "Oct 2017 — Aug 2018",
    role: "Volunteer",
    organization:
      "SAMI Volunteer organization, Hanoi University of Science and Technology",
  },
];
