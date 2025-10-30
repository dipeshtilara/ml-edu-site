'use client'

import React, { useState, useMemo } from 'react'
import Link from 'next/link'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Brain, Search, Filter, Play, Code, FileText } from 'lucide-react'
import projectsData from '@/data/projects.json'

export default function ProjectsPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedDifficulty, setSelectedDifficulty] = useState('All')
  const [selectedUnit, setSelectedUnit] = useState('All')

  // Filter projects based on search and filters
  const filteredProjects = useMemo(() => {
    return projectsData.filter(project => {
      const matchesSearch = searchQuery === '' || 
        project.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        project.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        project.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase())) ||
        project.cbse_unit.toLowerCase().includes(searchQuery.toLowerCase())
      
      const matchesDifficulty = selectedDifficulty === 'All' || project.difficulty === selectedDifficulty
      
      const matchesUnit = selectedUnit === 'All' || project.cbse_unit.includes(selectedUnit)
      
      return matchesSearch && matchesDifficulty && matchesUnit
    })
  }, [searchQuery, selectedDifficulty, selectedUnit])

  // Get unique values for filters
  const difficulties = ['All', ...Array.from(new Set(projectsData.map(p => p.difficulty)))]
  const units = ['All', 'Supervised Learning', 'Unsupervised Learning', 'Deep Learning', 'Ensemble Learning', 'Probabilistic Learning']

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return 'bg-green-100 text-green-800 border-green-200'
      case 'Intermediate': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'Advanced': return 'bg-red-100 text-red-800 border-red-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const getUnitColor = (unit: string) => {
    if (unit.includes('Supervised')) return 'bg-blue-50 text-blue-700'
    if (unit.includes('Unsupervised')) return 'bg-purple-50 text-purple-700'
    if (unit.includes('Deep Learning')) return 'bg-indigo-50 text-indigo-700'
    if (unit.includes('Ensemble')) return 'bg-orange-50 text-orange-700'
    return 'bg-gray-50 text-gray-700'
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header Section */}
      <section className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white py-16">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-4xl mx-auto">
            <div className="flex justify-center mb-6">
              <Brain className="h-16 w-16" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-6">
              20 Comprehensive AI Projects
            </h1>
            <p className="text-xl text-blue-100 mb-8">
              Complete machine learning implementations with 300+ lines of code each. 
              Run demos, view source code, and export to Word format.
            </p>
            <div className="flex flex-wrap justify-center gap-4 text-sm">
              <div className="flex items-center gap-2 bg-white/10 px-3 py-1 rounded-full">
                <Code className="h-4 w-4" />
                <span>300+ Lines Each</span>
              </div>
              <div className="flex items-center gap-2 bg-white/10 px-3 py-1 rounded-full">
                <Play className="h-4 w-4" />
                <span>Interactive Demos</span>
              </div>
              <div className="flex items-center gap-2 bg-white/10 px-3 py-1 rounded-full">
                <FileText className="h-4 w-4" />
                <span>Export to Word</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Search and Filters */}
      <section className="py-8 bg-white border-b">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row gap-4 items-center">
            {/* Search */}
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
              <Input
                type="text"
                placeholder="Search projects, topics, or CBSE units..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 w-full"
              />
            </div>
            
            {/* Filters */}
            <div className="flex gap-4 items-center">
              <Filter className="h-5 w-5 text-gray-500" />
              
              {/* Difficulty Filter */}
              <select 
                value={selectedDifficulty}
                onChange={(e) => setSelectedDifficulty(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {difficulties.map(difficulty => (
                  <option key={difficulty} value={difficulty}>{difficulty}</option>
                ))}
              </select>
              
              {/* Unit Filter */}
              <select 
                value={selectedUnit}
                onChange={(e) => setSelectedUnit(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {units.map(unit => (
                  <option key={unit} value={unit}>{unit}</option>
                ))}
              </select>
            </div>
          </div>
          
          {/* Results count */}
          <div className="mt-4 text-sm text-gray-600">
            Showing {filteredProjects.length} of {projectsData.length} projects
          </div>
        </div>
      </section>

      {/* Projects Grid */}
      <section className="py-12">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          {filteredProjects.length === 0 ? (
            <div className="text-center py-12">
              <Brain className="h-16 w-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-xl font-medium text-gray-900 mb-2">No projects found</h3>
              <p className="text-gray-600">Try adjusting your search criteria or filters.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {filteredProjects.map((project) => (
                <Card key={project.id} className="hover:shadow-lg transition-all duration-300 border-0 shadow-md">
                  <CardHeader className="pb-4">
                    <div className="flex items-start justify-between mb-3">
                      <Badge className={`${getDifficultyColor(project.difficulty)} border text-xs font-medium`}>
                        {project.difficulty}
                      </Badge>
                      <div className="text-right text-xs text-gray-500">
                        Project #{project.id}
                      </div>
                    </div>
                    
                    <CardTitle className="text-lg leading-tight mb-2">
                      {project.title}
                    </CardTitle>
                    
                    <div className={`text-xs px-2 py-1 rounded-md ${getUnitColor(project.cbse_unit)} font-medium`}>
                      {project.cbse_unit}
                    </div>
                  </CardHeader>
                  
                  <CardContent className="space-y-4">
                    <CardDescription className="text-sm text-gray-600 leading-relaxed">
                      {project.description}
                    </CardDescription>
                    
                    {/* Learning Objectives */}
                    <div>
                      <h4 className="text-sm font-medium text-gray-900 mb-2">Learning Objectives:</h4>
                      <ul className="text-xs text-gray-600 space-y-1">
                        {project.objectives.slice(0, 2).map((objective, index) => (
                          <li key={index} className="flex items-start">
                            <span className="text-blue-500 mr-2">•</span>
                            <span>{objective}</span>
                          </li>
                        ))}
                        {project.objectives.length > 2 && (
                          <li className="text-gray-500 text-xs">+{project.objectives.length - 2} more objectives</li>
                        )}
                      </ul>
                    </div>
                    
                    {/* Tags */}
                    <div className="flex flex-wrap gap-1">
                      {project.tags.slice(0, 3).map((tag) => (
                        <Badge key={tag} variant="outline" className="text-xs px-2 py-1">
                          {tag}
                        </Badge>
                      ))}
                      {project.tags.length > 3 && (
                        <Badge variant="outline" className="text-xs px-2 py-1 text-gray-500">
                          +{project.tags.length - 3}
                        </Badge>
                      )}
                    </div>
                    
                    {/* Action Buttons */}
                    <div className="flex gap-2 pt-2">
                      <Link href={`/projects/${project.slug}`} className="flex-1">
                        <Button className="w-full text-sm">
                          <Play className="mr-2 h-4 w-4" />
                          Run Demo
                        </Button>
                      </Link>
                      <Link href={`/projects/${project.slug}#code`}>
                        <Button variant="outline" size="sm">
                          <Code className="h-4 w-4" />
                        </Button>
                      </Link>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Start Learning?</h2>
          <p className="text-xl text-indigo-100 mb-8 max-w-2xl mx-auto">
            Each project includes complete source code, interactive demonstrations, 
            and detailed explanations aligned with CBSE Class 12 curriculum.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/cbse-mapping">
              <Button size="lg" variant="secondary">
                View CBSE Mapping
              </Button>
            </Link>
            <Link href="/about">
              <Button size="lg" variant="outline" className="border-white text-white hover:bg-white hover:text-indigo-600">
                Learn More
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}