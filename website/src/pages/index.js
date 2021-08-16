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
          <Button href={'/torchTS/docs/'}>Get Started</Button>
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
              and engineering
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
              content: '## Why Time Series? \n - Time series data modeling has broad significance in public health, finance and engineering. \n - Traditional time series methods from statistics often rely on strong modeling assumptions, or are computationally expensive. \n - Given the rise of large-scale sensing data and significant advances in deep learning, the goal of the project is to develop an efficient and user-friendly deep learning library that would benefit the entire research community and beyond.',
              image: `${baseUrl}img/time-series-graph.png`,
              imageAlt: 'The problem (picture of a question mark)',
              imageAlign: 'right',
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
            image: `${baseUrl}img/why.png`,
            imageAlign: 'left',
            imageAlt: 'The solution (picture of a star)',
            content: '## Why TorchTS? \n - Existing time series analysis libraries include [statsmodels](https://www.statsmodels.org/stable/index.html), [sktime](https://github.com/alan-turing-institute/sktime). However, these libraries only include traditional statistics tools such as ARMA or ARIMA, which do not have the state-of-the-art forecasting tools based on deep learning. \n - [GluonTS](https://ts.gluon.ai/) is an open-source time series library developed by Amazon AWS, but is based on MXNet. \n - [Pyro](https://pyro.ai/) is a probabilistic programming framework based on PyTorch, but is not focused on time series forecasting.',
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
            content: 'Easy to use with model.predict',
            image: `${baseUrl}img/puzzle.png`,
            imageAlign: 'top',
            title: 'User Friendly',
          },
          {
            content: 'Easily Scalable Library',
            image: `${baseUrl}img/scalable.png`,
            imageAlign: 'top',
            title: 'Scalable',
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
