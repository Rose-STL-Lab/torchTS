/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import { GridBlock } from '../components/GridBlock'
import { Container } from '../components/Container'
import Layout from '@theme/Layout'

const HomeSplash = (props) => {
  const { language = '' } = props
  const { siteConfig } = useDocusaurusContext()
  const { baseUrl, customFields } = siteConfig
  const docsPart = `${customFields.docsPath ? `${customFields.docsPath}/` : ''}`
  const langPart = `${language ? `${language}/` : ''}`
  const docUrl = (doc) => `${baseUrl}${docsPart}${langPart}${doc}`

  const SplashContainer = (props) => (
    <div className="homeContainer">
      <div className="homeSplashFade">
        <div className="wrapper homeWrapper">{props.children}</div>
      </div>
    </div>
  )

  const Logo = (props) => (
    <div className="projectLogo">
      <img src={props.img_src} alt="Project Logo" />
    </div>
  )

  const ProjectTitle = () => (
    <div>
      <h2 className="projectTitle">{siteConfig.title}</h2>
      <div className="projectTaglineWrapper">
        <p className="projectTagline">{siteConfig.tagline}</p>
      </div>
    </div>
  )

  const Button = (props) => (
    <a
      className="button button--primary button--outline"
      href={props.href}
      target={props.target}
    >
      {props.children}
    </a>
  )

  return (
    <SplashContainer>
      <Logo img_src={`${baseUrl}img/torchTS_logo.png`} />
      <div className="inner">
        <ProjectTitle siteConfig={siteConfig} />
        <div className="pluginWrapper buttonWrapper">
          <Button href={'/docs/'}>Get Started</Button>
        </div>
      </div>
    </SplashContainer>
  )
}

export default class Index extends React.Component {
  render() {
    const { config: siteConfig, language = '' } = this.props
    const { baseUrl } = siteConfig

    const Block = (props) => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}
      >
        <GridBlock
          align={props.align || 'center'}
          imageAlign={props.imageAlign || 'center'}
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    )

    const FeatureCallout = () => (
      <Container className="" background={'light'} padding={['top', 'bottom']}>
        <div style={{ textAlign: 'center' }}>
          <p>
            <i>
              Time series data modeling has broad significance in public health, finance
              and engineering.
            </i>
          </p>
        </div>
      </Container>
    )

    const Problem = () => (
      <React.Fragment>
        <Block background={'light'} align="left">
          {[
            {
              title: '',
              content: '## The Problem \n - problem statement',
              image: `${baseUrl}img/interrobang-128x128.png`,
              imageAlt: 'The problem (picture of a question mark)',
              imageAlign: 'left',
            },
          ]}
        </Block>
      </React.Fragment>
    )

    const Solution = () => [
      <Block background={null} align="left">
        {[
          {
            title: '',
            image: `${baseUrl}img/star-128x128.png`,
            imageAlign: 'right',
            imageAlt: 'The solution (picture of a star)',
            content: '## The Solution \n The solution',
          },
        ]}
      </Block>,
    ]

    const Features = () => (
      <Block layout="twoColumn">
        {[
          {
            content: 'Library built on pytorch',
            image: `${baseUrl}img/pytorch-logo.png`,
            imageAlign: 'top',
            title: 'Built On Pytorch',
          },
          {
            content: 'Another Tagline',
            image: `${baseUrl}img/check-128x128.png`,
            imageAlign: 'top',
            title: 'Another Tagline',
          },
          {
            content: 'Another Tagline',
            image: `${baseUrl}img/tada-128x128.png`,
            imageAlign: 'top',
            title: 'Another Tagline',
          },
        ]}
      </Block>
    )

    return (
      <Layout permalink="/" title={siteConfig.title} description={siteConfig.tagline}>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="mainContainer">
          <FeatureCallout />
          <Features />
          <Problem />
          <Solution />
        </div>
      </Layout>
    )
  }
}
