'use client'

import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { BookOpen, Brain, Target, CheckCircle, ArrowRight } from 'lucide-react'
import Link from 'next/link'
import projectsData from '@/data/projects.json'

export default function CBSEMappingPage() {
  // Group projects by CBSE units
  const unitMapping = projectsData.reduce((acc: any, project) => {
    const unit = project.cbse_unit
    if (!acc[unit]) {
      acc[unit] = []
    }
    acc[unit].push(project)
    return acc
  }, {})

  const units = [
    {
      name: 'Supervised Learning - Regression',
      description: 'Algorithms that learn from labeled data to predict continuous values',
      color: 'bg-blue-50 text-blue-700 border-blue-200',
      icon: '📈'
    },
    {
      name: 'Supervised Learning - Classification', 
      description: 'Algorithms that learn from labeled data to predict categories',
      color: 'bg-green-50 text-green-700 border-green-200',
      icon: '🎯'
    },
    {
      name: 'Supervised Learning - Tree Algorithms',
      description: 'Decision-based algorithms that create interpretable rules',
      color: 'bg-purple-50 text-purple-700 border-purple-200', 
      icon: '🌳'
    },
    {
      name: 'Ensemble Learning',
      description: 'Methods that combine multiple models for better performance',
      color: 'bg-orange-50 text-orange-700 border-orange-200',
      icon: '🔗'
    },
    {
      name: 'Support Vector Machines',
      description: 'Powerful algorithms for both classification and regression',
      color: 'bg-red-50 text-red-700 border-red-200',
      icon: '⚡'
    },
    {
      name: 'Unsupervised Learning - Clustering',
      description: 'Algorithms that find hidden patterns in unlabeled data',
      color: 'bg-indigo-50 text-indigo-700 border-indigo-200',
      icon: '🔍'
    },
    {
      name: 'Deep Learning - Neural Networks',
      description: 'Multi-layer networks inspired by biological neural systems',
      color: 'bg-pink-50 text-pink-700 border-pink-200',
      icon: '🧠'
    },
    {
      name: 'Deep Learning - Convolutional Networks',
      description: 'Specialized neural networks for image and spatial data',
      color: 'bg-cyan-50 text-cyan-700 border-cyan-200',
      icon: '👁️'
    },
    {
      name: 'Probabilistic Learning',
      description: 'Algorithms based on probability theory and Bayes theorem',
      color: 'bg-yellow-50 text-yellow-700 border-yellow-200',
      icon: '🎲'
    },
    {
      name: 'Instance-based Learning',
      description: 'Algorithms that make decisions based on similar examples',
      color: 'bg-teal-50 text-teal-700 border-teal-200',
      icon: '📊'
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
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <section className="bg-gradient-to-r from-purple-600 to-indigo-700 text-white py-16">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-4xl mx-auto">
            <div className="flex justify-center mb-6">
              <BookOpen className="h-16 w-16" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-6">
              CBSE Class 12 AI Curriculum Mapping
            </h1>
            <p className="text-xl text-purple-100 mb-8">
              Complete alignment of our 20 AI projects with CBSE Class 12 Artificial Intelligence 
              curriculum units and learning objectives.
            </p>
            <div className="flex flex-wrap justify-center gap-4 text-sm">
              <div className="flex items-center gap-2 bg-white/10 px-3 py-1 rounded-full">
                <Target className="h-4 w-4" />
                <span>Curriculum Aligned</span>
              </div>
              <div className="flex items-center gap-2 bg-white/10 px-3 py-1 rounded-full">
                <CheckCircle className="h-4 w-4" />
                <span>CBSE Standards</span>
              </div>
              <div className="flex items-center gap-2 bg-white/10 px-3 py-1 rounded-full">
                <Brain className="h-4 w-4" />
                <span>Practical Learning</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Overview Statistics */}
      <section className="py-12 bg-white">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card className="text-center">
              <CardContent className="pt-6">
                <div className="text-3xl font-bold text-blue-600 mb-2">20</div>
                <div className="text-sm text-gray-600">Total Projects</div>
              </CardContent>
            </Card>
            <Card className="text-center">
              <CardContent className="pt-6">
                <div className="text-3xl font-bold text-green-600 mb-2">{Object.keys(unitMapping).length}</div>
                <div className="text-sm text-gray-600">CBSE Units Covered</div>
              </CardContent>
            </Card>
            <Card className="text-center">
              <CardContent className="pt-6">
                <div className="text-3xl font-bold text-purple-600 mb-2">6000+</div>
                <div className="text-sm text-gray-600">Lines of Code</div>
              </CardContent>
            </Card>
            <Card className="text-center">
              <CardContent className="pt-6">
                <div className="text-3xl font-bold text-orange-600 mb-2">100%</div>
                <div className="text-sm text-gray-600">Curriculum Coverage</div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Unit Mapping */}
      <section className="py-12">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Curriculum Unit Mapping
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Each project is carefully designed to address specific learning objectives 
              from the CBSE Class 12 AI curriculum.
            </p>
          </div>

          <div className="space-y-8">
            {units.map((unit) => {
              const projects = unitMapping[unit.name] || []
              
              return (
                <Card key={unit.name} className="overflow-hidden">
                  <CardHeader className={`${unit.color} border-b`}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <span className="text-2xl">{unit.icon}</span>
                        <div>
                          <CardTitle className="text-xl">{unit.name}</CardTitle>
                          <CardDescription className="mt-1 text-sm opacity-80">
                            {unit.description}
                          </CardDescription>
                        </div>
                      </div>
                      <Badge variant="outline" className="border-current">
                        {projects.length} Project{projects.length !== 1 ? 's' : ''}
                      </Badge>
                    </div>
                  </CardHeader>
                  
                  <CardContent className="p-6">
                    {projects.length > 0 ? (
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {projects.map((project: any) => (
                          <div key={project.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                            <div className="flex items-start justify-between mb-3">
                              <Badge className={`${getDifficultyColor(project.difficulty)} text-xs`}>
                                {project.difficulty}
                              </Badge>
                              <span className="text-xs text-gray-500">#{project.id}</span>
                            </div>
                            
                            <h4 className="font-semibold text-gray-900 mb-2 leading-tight">
                              {project.title}
                            </h4>
                            
                            <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                              {project.description}
                            </p>
                            
                            <div className="flex flex-wrap gap-1 mb-3">
                              {project.tags.slice(0, 2).map((tag: string) => (
                                <Badge key={tag} variant="outline" className="text-xs">
                                  {tag}
                                </Badge>
                              ))}
                              {project.tags.length > 2 && (
                                <Badge variant="outline" className="text-xs text-gray-500">
                                  +{project.tags.length - 2}
                                </Badge>
                              )}
                            </div>
                            
                            <Link href={`/projects/${project.slug}`}>
                              <Button variant="outline" size="sm" className="w-full">
                                View Project
                                <ArrowRight className="ml-2 h-3 w-3" />
                              </Button>
                            </Link>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-8">
                        <Brain className="h-12 w-12 text-gray-400 mx-auto mb-3" />
                        <p className="text-gray-600">No projects available for this unit yet.</p>
                        <p className="text-sm text-gray-500">More projects coming soon!</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </div>
      </section>

      {/* Learning Objectives */}
      <section className="py-16 bg-white">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">
              Key Learning Objectives
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <CheckCircle className="h-5 w-5 mr-2 text-green-600" />
                    Theoretical Understanding
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2 text-sm text-gray-600">
                    <li>• Mathematical foundations of AI algorithms</li>
                    <li>• Statistical concepts and probability theory</li>
                    <li>• Optimization techniques and cost functions</li>
                    <li>• Performance evaluation metrics</li>
                  </ul>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <CheckCircle className="h-5 w-5 mr-2 text-blue-600" />
                    Practical Implementation
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2 text-sm text-gray-600">
                    <li>• Complete algorithm implementation from scratch</li>
                    <li>• Data preprocessing and feature engineering</li>
                    <li>• Model training and hyperparameter tuning</li>
                    <li>• Result visualization and interpretation</li>
                  </ul>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <CheckCircle className="h-5 w-5 mr-2 text-purple-600" />
                    Real-world Applications
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2 text-sm text-gray-600">
                    <li>• Healthcare and medical diagnosis</li>
                    <li>• Financial analysis and prediction</li>
                    <li>• Customer behavior analysis</li>
                    <li>• Natural language processing tasks</li>
                  </ul>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <CheckCircle className="h-5 w-5 mr-2 text-orange-600" />
                    Professional Skills
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2 text-sm text-gray-600">
                    <li>• Software engineering best practices</li>
                    <li>• Documentation and code organization</li>
                    <li>• Project presentation and reporting</li>
                    <li>• Ethical AI considerations</li>
                  </ul>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold mb-4">Start Your AI Learning Journey</h2>
          <p className="text-xl text-indigo-100 mb-8 max-w-2xl mx-auto">
            Explore our comprehensive AI projects designed specifically for CBSE Class 12 students.
            Each project provides hands-on experience with real-world applications.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/projects">
              <Button size="lg" variant="secondary">
                <Brain className="mr-2 h-5 w-5" />
                Explore All Projects
              </Button>
            </Link>
            <Link href="/about">
              <Button size="lg" variant="outline" className="border-white text-white hover:bg-white hover:text-indigo-600">
                <BookOpen className="mr-2 h-5 w-5" />
                Learn More
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}