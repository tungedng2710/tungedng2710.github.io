const dashboard = document.querySelector<HTMLElement>("#career-dashboard");

if (dashboard) {
  const panels = Array.from(
    dashboard.querySelectorAll<HTMLElement>("[data-panel]"),
  );
  const navLinks = Array.from(
    dashboard.querySelectorAll<HTMLAnchorElement>("[data-section-link]"),
  );
  const search = dashboard.querySelector<HTMLInputElement>("#resume-search")!;
  const feedback = dashboard.querySelector<HTMLElement>("#search-feedback")!;
  const searchStatus = dashboard.querySelector<HTMLElement>("#search-status")!;
  const emptySearch = dashboard.querySelector<HTMLElement>("#empty-search")!;
  const currentSection =
    dashboard.querySelector<HTMLElement>("#current-section")!;
  const filters = new Map<string, string>();
  let activePanel = "overview";
  let detailsBeforeSearch: Map<HTMLDetailsElement, boolean> | null = null;

  // Record text once, including collapsed details, so every part of the CV is searchable.
  const records = Array.from(
    dashboard.querySelectorAll<HTMLElement>("[data-record]"),
  ).map((element) => ({
    element,
    panel: element.closest<HTMLElement>("[data-panel]")!,
    text: (element.textContent ?? "")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLocaleLowerCase(),
  }));

  function render() {
    const query = search.value
      .trim()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLocaleLowerCase();
    const terms = query.split(/\s+/).filter(Boolean);
    dashboard!.classList.toggle("is-searching", Boolean(query));
    feedback.hidden = !query;
    let matches = 0;

    if (query && !detailsBeforeSearch) {
      detailsBeforeSearch = new Map(
        Array.from(
          dashboard!.querySelectorAll<HTMLDetailsElement>("details"),
        ).map((detail) => [detail, detail.open]),
      );
    } else if (!query && detailsBeforeSearch) {
      detailsBeforeSearch.forEach((open, detail) => {
        detail.open = open;
      });
      detailsBeforeSearch = null;
    }

    for (const record of records) {
      const category = filters.get(record.panel.id) ?? "All";
      const visible = query
        ? terms.every((term) => record.text.includes(term))
        : category === "All" || record.element.dataset.category === category;
      record.element.hidden = !visible;
      if (query && visible) {
        matches++;
        record.element
          .querySelectorAll<HTMLDetailsElement>("details")
          .forEach((detail) => {
            detail.open = true;
          });
      }
    }

    for (const panel of panels) {
      panel.hidden = query
        ? !records.some(
            (record) => record.panel === panel && !record.element.hidden,
          )
        : panel.id !== activePanel;
    }

    navLinks.forEach((link) => {
      const selected = !query && link.dataset.sectionLink === activePanel;
      link.setAttribute("aria-selected", String(selected));
      link.tabIndex = link.dataset.sectionLink === activePanel ? 0 : -1;
    });
    emptySearch.hidden = !query || matches > 0;
    searchStatus.textContent = `${matches} ${matches === 1 ? "result" : "results"} for “${search.value.trim()}”`;
    currentSection.textContent = query
      ? "Search results"
      : (navLinks
          .find((link) => link.dataset.sectionLink === activePanel)
          ?.querySelector("span")?.textContent ?? "Overview");
  }

  function setFilter(panelId: string, category: string) {
    filters.set(panelId, category);
    dashboard!
      .querySelectorAll<HTMLButtonElement>(
        `[data-filter-group="${panelId}"] [data-filter]`,
      )
      .forEach((button) => {
        button.setAttribute(
          "aria-pressed",
          String(button.dataset.filter === category),
        );
      });
  }

  function navigate(panelId: string, updateHash = true) {
    if (!panels.some((panel) => panel.id === panelId)) panelId = "overview";
    activePanel = panelId;
    search.value = "";
    render();
    if (updateHash) history.replaceState(null, "", `#${panelId}`);
  }

  const tablist = dashboard.querySelector<HTMLElement>(".dashboard-nav")!;
  tablist.setAttribute("role", "tablist");
  tablist.setAttribute(
    "aria-orientation",
    matchMedia("(max-width: 720px)").matches ? "horizontal" : "vertical",
  );
  matchMedia("(max-width: 720px)").addEventListener("change", (event) => {
    tablist.setAttribute(
      "aria-orientation",
      event.matches ? "horizontal" : "vertical",
    );
  });
  navLinks.forEach((link) => {
    link.setAttribute("role", "tab");
    link.setAttribute("aria-controls", link.dataset.sectionLink!);
  });
  panels.forEach((panel) => {
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", `nav-${panel.id}`);
    panel.tabIndex = 0;
  });

  dashboard.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const link = target?.closest<HTMLAnchorElement>(
      "[data-section-link], [data-go]",
    );
    if (
      link &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.shiftKey &&
      !event.altKey
    ) {
      event.preventDefault();
      const panelId = link.dataset.sectionLink ?? link.dataset.go!;
      // Overview shortcuts always reveal the requested content, even after an earlier filter.
      if (link.dataset.go) setFilter(panelId, link.dataset.toolkit ?? "All");
      navigate(panelId);
      const role = link.dataset.role;
      if (role !== undefined) {
        const detail = dashboard!.querySelector<HTMLDetailsElement>(
          `#role-${role}`,
        );
        if (detail) {
          detail.open = true;
          detail.querySelector("summary")?.focus({ preventScroll: true });
          detail.scrollIntoView({ block: "center", behavior: "instant" });
        }
      } else {
        if (link.dataset.go)
          panels
            .find((panel) => panel.id === panelId)
            ?.focus({ preventScroll: true });
        dashboard!.scrollIntoView({ block: "start", behavior: "instant" });
      }
    }
    const filter = target?.closest<HTMLButtonElement>("[data-filter]");
    if (filter) {
      const group = filter.closest<HTMLElement>("[data-filter-group]")!;
      setFilter(group.dataset.filterGroup!, filter.dataset.filter!);
      render();
    }
  });

  tablist.addEventListener("keydown", (event) => {
    const index = navLinks.indexOf(document.activeElement as HTMLAnchorElement);
    if (index < 0) return;
    const horizontal =
      tablist.getAttribute("aria-orientation") === "horizontal";
    const nextKey = horizontal ? "ArrowRight" : "ArrowDown";
    const previousKey = horizontal ? "ArrowLeft" : "ArrowUp";
    let next: number;
    if (event.key === nextKey) next = (index + 1) % navLinks.length;
    else if (event.key === previousKey)
      next = (index - 1 + navLinks.length) % navLinks.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = navLinks.length - 1;
    else return;
    event.preventDefault();
    navLinks[next].focus();
    navigate(navLinks[next].dataset.sectionLink!);
  });

  search.addEventListener("input", render);
  const clearSearch = () => {
    search.value = "";
    render();
    search.focus();
  };
  dashboard
    .querySelector("#clear-search")
    ?.addEventListener("click", clearSearch);
  dashboard
    .querySelector("[data-reset-search]")
    ?.addEventListener("click", clearSearch);
  document.addEventListener("keydown", (event) => {
    const target = event.target instanceof HTMLElement ? event.target : null;
    if (
      event.key === "/" &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey &&
      !target?.closest("input, textarea, select, [contenteditable]")
    ) {
      event.preventDefault();
      search.focus();
    }
    if (
      event.key === "Escape" &&
      document.activeElement === search &&
      search.value
    )
      clearSearch();
  });
  window.addEventListener("hashchange", () =>
    navigate(location.hash.slice(1), false),
  );

  let copyTimer: ReturnType<typeof setTimeout>;
  const copyButton = dashboard.querySelector<HTMLButtonElement>(".copy-email")!;
  copyButton.addEventListener("click", async () => {
    const status = dashboard!.querySelector<HTMLElement>("#copy-status")!;
    clearTimeout(copyTimer);
    try {
      await navigator.clipboard.writeText(copyButton.dataset.email!);
      status.textContent = "Email copied. Let’s talk!";
      copyButton.setAttribute("aria-label", "Email address copied");
    } catch {
      status.textContent = copyButton.dataset.email!;
      copyButton.setAttribute("aria-label", "Copy email address");
    }
    copyTimer = setTimeout(() => {
      status.textContent = "";
      copyButton.setAttribute("aria-label", "Copy email address");
    }, 4000);
  });

  // Include collapsed contributions when visitors print the dashboard.
  let closedDetails: HTMLDetailsElement[] = [];
  window.addEventListener("beforeprint", () => {
    closedDetails = Array.from(
      dashboard!.querySelectorAll<HTMLDetailsElement>("details:not([open])"),
    );
    closedDetails.forEach((detail) => {
      detail.open = true;
    });
  });
  window.addEventListener("afterprint", () => {
    closedDetails.forEach((detail) => {
      detail.open = false;
    });
  });

  navigate(location.hash.slice(1), false);
  dashboard.classList.add("resume-ready");
}
