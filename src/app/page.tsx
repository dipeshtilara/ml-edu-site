'use client'

import React from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Brain, Code, FileText, Play, BookOpen, Award, Users, Download } from 'lucide-react'

export default function HomePage() {
  const features = [
    {
      icon: Code,
      title: "20+ AI Projects",
      description: "Comprehensive machine learning projects covering the entire CBSE Class 12 AI curriculum with 300+ lines of code each."
    },
    {
      icon: Play,
      title: "Interactive Demos",
      description: "Run Python code directly in your browser using Pyodide. No installation required - just click and execute."
    },
    {
      icon: FileText,
      title: "Export to Word",
      description: "Download complete projects with code, outputs, and screenshots as Word documents for easy submission."
    },
    {
      icon: BookOpen,
      title: "CBSE Aligned",
      description: "Every project is mapped to specific CBSE units with learning objectives and curriculum alignment."
    },
    {
      icon: Award,
      title: "Beginner to Advanced",
      description: "Projects range from basic regression to advanced neural networks, suitable for all skill levels."
    },
    {
      icon: Users,
      title: "Student Friendly",
      description: "Clear explanations, visual outputs, and dependency notes make complex AI concepts accessible."
    }
  ]

  const popularProjects = [
    {
      id: 1,
      title: "Linear Regression: Student Performance",
      description: "Predict exam scores using study hours and attendance data",
      difficulty: "Beginner",
      tags: ["regression", "statistics"]
    },
    {
      id: 8,
      title: "Neural Network: Image Classification",
      description: "Build multi-layer networks with backpropagation",
      difficulty: "Advanced",
      tags: ["deep-learning", "neural-network"]
    },
    {
      id: 6,
      title: "K-Means: Customer Segmentation",
      description: "Cluster customers based on purchasing behavior",
      difficulty: "Beginner",
      tags: ["clustering", "unsupervised"]
    }
  ]

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return 'bg-green-100 text-green-800'
      case 'Intermediate': return 'bg-yellow-100 text-yellow-800'
      case 'Advanced': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 py-20">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-4xl mx-auto">
            <div className="flex justify-center mb-6">
              <Brain className="h-16 w-16 text-primary" />
            </div>
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              AI Learning Hub for
              <span className="text-primary block">CBSE Class 12</span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
              Master Artificial Intelligence with 20 comprehensive, runnable projects. 
              Each project contains 300+ lines of code, covers CBSE curriculum, and exports to Word format.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/projects">
                <Button size="lg" className="w-full sm:w-auto">
                  <Play className="mr-2 h-5 w-5" />
                  Start Learning
                </Button>
              </Link>
              <Link href="/cbse-mapping">
                <Button size="lg" variant="outline" className="w-full sm:w-auto">
                  <BookOpen className="mr-2 h-5 w-5" />
                  View Syllabus
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Everything You Need to Excel in AI
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Our platform provides comprehensive resources aligned with CBSE Class 12 AI curriculum
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <Card key={index} className="hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-primary/10 rounded-lg">
                        <Icon className="h-6 w-6 text-primary" />
                      </div>
                      <CardTitle className="text-lg">{feature.title}</CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-base">
                      {feature.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </div>
      </section>

      {/* Popular Projects Section */}
      <section className="py-20 bg-gray-50">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Popular Projects
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Start with these student-favorite projects that demonstrate key AI concepts
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {popularProjects.map((project) => (
              <Card key={project.id} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <Badge className={getDifficultyColor(project.difficulty)}>
                      {project.difficulty}
                    </Badge>
                    <Brain className="h-5 w-5 text-primary" />
                  </div>
                  <CardTitle className="text-lg">{project.title}</CardTitle>
                  <CardDescription>{project.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {project.tags.map((tag) => (
                      <Badge key={tag} variant="outline" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                  <Link href={`/projects/${project.id}`}>
                    <Button className="w-full">
                      <Play className="mr-2 h-4 w-4" />
                      Run Project
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            ))}
          </div>
          
          <div className="text-center mt-12">
            <Link href="/projects">
              <Button size="lg" variant="outline">
                View All 20 Projects
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-primary">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="max-w-3xl mx-auto text-white">
            <h2 className="text-3xl md:text-4xl font-bold mb-6">
              Ready to Master AI?
            </h2>
            <p className="text-xl mb-8 opacity-90">
              Join thousands of CBSE students who are already building amazing AI projects. 
              Start your journey today with our comprehensive learning platform.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/projects">
                <Button size="lg" variant="secondary" className="w-full sm:w-auto">
                  <Code className="mr-2 h-5 w-5" />
                  Explore Projects
                </Button>
              </Link>
              <Link href="/about">
                <Button size="lg" variant="outline" className="w-full sm:w-auto border-white text-white hover:bg-white hover:text-primary">
                  <Download className="mr-2 h-5 w-5" />
                  Learn More
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}