const lightCodeTheme = require("prism-react-renderer/themes/github");
const darkCodeTheme = require("prism-react-renderer/themes/dracula");

/** @type {import('@docusaurus/types').DocusaurusConfig} */
module.exports = {
  title: "TorchTS",
  tagline: "Time series forecasting with PyTorch",
  url: "https://rose-stl-lab.github.io",
  baseUrl: "/torchTS/",
  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",
  favicon: "img/logo2.png",
  scripts: [
    "https://buttons.github.io/buttons.js",
    "https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.0/clipboard.min.js",
  ],
  stylesheets: [
    "https://fonts.googleapis.com/css?family=IBM+Plex+Mono:500,700|Source+Code+Pro:500,700|Source+Sans+Pro:400,400i,700",
  ],

  organizationName: "Rose-STL-Lab", // Usually your GitHub org/user name.
  projectName: "torchTS", // Usually your repo name.
  themeConfig: {
    // colorMode: {
    // defaultMode: "light",
    // disableSwitch: true,
    // },
    navbar: {
      title: "TorchTS",
      logo: {
        alt: "My Site Logo",
        src: "img/logo2.png",
      },
      items: [
        {
          type: "doc",
          docId: "intro",
          position: "left",
          label: "Docs",
        },
        {
          href: "https://github.com/Rose-STL-Lab/torchts",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      links: [
        {
          title: "Docs",
          items: [
            {
              label: "Getting Started",
              to: "docs",
            },
            // {
            // label: 'Tutorials',
            // to: '/tutorials',
            // },
            // {
            // label: 'API',
            // to: '/api',
            // },
          ],
        },
        {
          title: "Community",
          items: [
            {
              label: "Slack",
              href: "https://github.com/Rose-Stl-Lab/torchTS",
            },
            {
              label: "Discord",
              href: "https://github.com/Rose-Stl-Lab/torchTS",
            },
          ],
        },
        {
          title: "More",
          items: [
            {
              html: `
 <a target="_blank" rel="noreferrer noopener" class="github-button"
 href="https://github.com/Rose-stl-lab/torchts"
 data-icon="octicon-star"
 data-count-href="/Rose-stl-lab/torchts/stargazers"
 data-show-count="true"
 data-count-aria-label="# stargazers on GitHub"
 aria-label="Star this project on GitHub">Star</a>
 `,
            },
            {
              label: "GitHub",
              href: "https://github.com/Rose-stl-lab/torchts",
            },
            {
              label: "Edit Docs on GitHub",
              href: "https://github.com/Rose-stl-lab/torchts/",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} TorchTS Team`,
      logo: {
        src: "img/octopus-128x128.png",
      },
    },
    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
    },
    fonts: {
      fontMain: ["Source Sans Pro", "sans-serif"],
      fontCode: ["IBM Plex Mono", "monospace"],
    },
  },
  presets: [
    [
      "@docusaurus/preset-classic",
      {
        docs: {
          sidebarPath: require.resolve("./sidebars.js"),
          // Please change this to your repo.
          editUrl: "https://github.com/Rose-STL-Lab/torchTS/edit/main/website/",
        },
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      },
    ],
  ],
};
