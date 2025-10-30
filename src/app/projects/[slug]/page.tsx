'use client'

import React, { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ArrowLeft, Play, Square, Download, Copy, FileText, Brain, Code2, CheckCircle, AlertCircle } from 'lucide-react'
import projectsData from '@/data/projects.json'
import { CodeViewer } from '@/components/project/CodeViewer'
import { PythonRunner } from '@/components/project/PythonRunner'
import { WordExporter } from '@/components/project/WordExporter'

export default function ProjectDetailPage() {
  const params = useParams()
  const [project, setProject] = useState<any>(null)
  const [activeTab, setActiveTab] = useState('overview')
  const [isRunning, setIsRunning] = useState(false)
  const [output, setOutput] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    const foundProject = projectsData.find(p => p.slug === params.slug)
    setProject(foundProject)
  }, [params.slug])

  const handleRunCode = async () => {
    if (!project) return
    
    setIsRunning(true)
    setOutput('')
    setError('')
    
    try {
      // This would integrate with Pyodide in a real implementation
      setOutput('Running AI project... This is a demo output.\n\nProject: ' + project.title + '\nStatus: Execution started\n\nDemo results would appear here in the full implementation.')
    } catch (err) {
      setError('Error running project: ' + err)
    } finally {
      setIsRunning(false)
    }
  }

  const handleStopCode = () => {
    setIsRunning(false)
    setOutput(prev => prev + '\n\nExecution stopped by user.')
  }

  const handleExportWord = async () => {
    if (!project) return
    
    try {
      // This would integrate with the WordExporter component
      alert('Word export functionality will be implemented with docx package and screenshot capture')
    } catch (err) {
      console.error('Export error:', err)
    }
  }

  if (!project) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Brain className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-2xl font-semibold text-gray-900 mb-2">Project Not Found</h2>
          <p className="text-gray-600 mb-6">The requested AI project could not be found.</p>
          <Link href="/projects">
            <Button>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Projects
            </Button>
          </Link>
        </div>
      </div>
    )
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return 'bg-green-100 text-green-800 border-green-200'
      case 'Intermediate': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'Advanced': return 'bg-red-100 text-red-800 border-red-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: FileText },
    { id: 'demo', label: 'Run Demo', icon: Play },
    { id: 'code', label: 'Source Code', icon: Code2 }
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/projects">
                <Button variant="ghost" size="sm">
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  All Projects
                </Button>
              </Link>
              <div className="h-6 border-l border-gray-300" />
              <Badge className={`${getDifficultyColor(project.difficulty)} border text-xs`}>
                {project.difficulty}
              </Badge>
              <span className="text-sm text-gray-500">Project #{project.id}</span>
            </div>
            
            <div className="flex items-center space-x-2">
              <Button onClick={handleExportWord} variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Export to Word
              </Button>
            </div>
          </div>
          
          <div className="mt-4">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">{project.title}</h1>
            <p className="text-lg text-gray-600">{project.description}</p>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white border-b">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{tab.label}</span>
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Main Content */}
            <div className="lg:col-span-2 space-y-6">
              {/* CBSE Unit */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Brain className="h-5 w-5 mr-2 text-blue-600" />
                    CBSE Curriculum Alignment
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-blue-900 mb-2">{project.cbse_unit}</h3>
                    <p className="text-blue-700 text-sm">
                      This project is specifically designed to cover the {project.cbse_unit} unit 
                      of the CBSE Class 12 Artificial Intelligence curriculum.
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* Learning Objectives */}
              <Card>
                <CardHeader>
                  <CardTitle>Learning Objectives</CardTitle>
                  <CardDescription>
                    What you'll learn by completing this project
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-3">
                    {project.objectives.map((objective: string, index: number) => (
                      <li key={index} className="flex items-start">
                        <CheckCircle className="h-5 w-5 text-green-500 mr-3 mt-0.5 flex-shrink-0" />
                        <span className="text-gray-700">{objective}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>

              {/* Project Features */}
              <Card>
                <CardHeader>
                  <CardTitle>Project Features</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-center space-x-3">
                      <div className="bg-green-100 p-2 rounded-lg">
                        <Code2 className="h-5 w-5 text-green-600" />
                      </div>
                      <div>
                        <h4 className="font-semibold">300+ Lines of Code</h4>
                        <p className="text-sm text-gray-600">Complete implementation from scratch</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-3">
                      <div className="bg-blue-100 p-2 rounded-lg">
                        <Play className="h-5 w-5 text-blue-600" />
                      </div>
                      <div>
                        <h4 className="font-semibold">Interactive Demo</h4>
                        <p className="text-sm text-gray-600">Run code directly in browser</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-3">
                      <div className="bg-purple-100 p-2 rounded-lg">
                        <FileText className="h-5 w-5 text-purple-600" />
                      </div>
                      <div>
                        <h4 className="font-semibold">Detailed Analysis</h4>
                        <p className="text-sm text-gray-600">Comprehensive output & metrics</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-3">
                      <div className="bg-orange-100 p-2 rounded-lg">
                        <Download className="h-5 w-5 text-orange-600" />
                      </div>
                      <div>
                        <h4 className="font-semibold">Export Ready</h4>
                        <p className="text-sm text-gray-600">Download as Word document</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Sidebar */}
            <div className="space-y-6">
              {/* Quick Info */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Project Info</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-900">Language</h4>
                    <p className="text-sm text-gray-600 mt-1">{project.language.toUpperCase()}</p>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-medium text-gray-900">Runtime</h4>
                    <p className="text-sm text-gray-600 mt-1">{project.runnable}</p>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-medium text-gray-900">Demo Type</h4>
                    <p className="text-sm text-gray-600 mt-1">{project.demo_type}</p>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-medium text-gray-900">Dataset</h4>
                    <p className="text-sm text-gray-600 mt-1">{project.dataset}</p>
                  </div>
                </CardContent>
              </Card>

              {/* Tags */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Topics Covered</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {project.tags.map((tag: string) => (
                      <Badge key={tag} variant="outline" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Quick Actions */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Quick Actions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Button 
                    onClick={() => setActiveTab('demo')} 
                    className="w-full"
                  >
                    <Play className="mr-2 h-4 w-4" />
                    Run Demo
                  </Button>
                  
                  <Button 
                    onClick={() => setActiveTab('code')} 
                    variant="outline" 
                    className="w-full"
                  >
                    <Code2 className="mr-2 h-4 w-4" />
                    View Code
                  </Button>
                  
                  <Button 
                    onClick={handleExportWord} 
                    variant="outline" 
                    className="w-full"
                  >
                    <Download className="mr-2 h-4 w-4" />
                    Export to Word
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'demo' && (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
            {/* Controls */}
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Play className="h-5 w-5 mr-2" />
                    Interactive Demo
                  </CardTitle>
                  <CardDescription>
                    Run the complete AI project and see real-time results
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex gap-3">
                    <Button 
                      onClick={handleRunCode} 
                      disabled={isRunning}
                      className="flex-1"
                    >
                      {isRunning ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                          Running...
                        </>
                      ) : (
                        <>
                          <Play className="mr-2 h-4 w-4" />
                          Run Project
                        </>
                      )}
                    </Button>
                    
                    {isRunning && (
                      <Button onClick={handleStopCode} variant="outline">
                        <Square className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                  
                  <div className="text-sm text-gray-600 bg-blue-50 p-3 rounded-lg">
                    <div className="flex items-start">
                      <AlertCircle className="h-4 w-4 text-blue-600 mr-2 mt-0.5 flex-shrink-0" />
                      <div>
                        <p className="font-medium text-blue-900 mb-1">Demo Environment</p>
                        <p className="text-blue-700">
                          This project runs using Pyodide (Python in the browser). 
                          All computations happen locally - no data is sent to servers.
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Dependencies Note */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Dependencies & Notes</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-semibold text-gray-900 mb-2">Key Dependencies:</h4>
                    <ul className="text-sm text-gray-600 space-y-1 mb-4">
                      <li>• <code className="bg-gray-200 px-1 rounded">math</code> - Mathematical operations</li>
                      <li>• <code className="bg-gray-200 px-1 rounded">random</code> - Data generation and sampling</li>
                      <li>• <code className="bg-gray-200 px-1 rounded">typing</code> - Type hints for better code</li>
                    </ul>
                    <p className="text-xs text-gray-500">
                      This implementation uses only Python built-in libraries for 
                      educational purposes and maximum compatibility.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Output */}
            <div>
              <Card className="h-full">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>Output</span>
                    {output && (
                      <Button 
                        onClick={() => navigator.clipboard.writeText(output)}
                        variant="outline" 
                        size="sm"
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="bg-black text-green-400 p-4 rounded-lg font-mono text-sm min-h-[400px] overflow-auto">
                    {isRunning && !output && (
                      <div className="flex items-center">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-green-400 mr-2" />
                        <span>Initializing Python environment...</span>
                      </div>
                    )}
                    {error && (
                      <div className="text-red-400">
                        <div className="flex items-center mb-2">
                          <AlertCircle className="h-4 w-4 mr-2" />
                          <span>Error:</span>
                        </div>
                        <pre className="whitespace-pre-wrap">{error}</pre>
                      </div>
                    )}
                    {output && (
                      <pre className="whitespace-pre-wrap">{output}</pre>
                    )}
                    {!output && !error && !isRunning && (
                      <div className="text-gray-500">
                        Click "Run Project" to execute the AI model and see results...
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'code' && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Source Code</span>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      <Copy className="h-4 w-4 mr-2" />
                      Copy All
                    </Button>
                    <Button variant="outline" size="sm">
                      <Download className="h-4 w-4 mr-2" />
                      Download
                    </Button>
                  </div>
                </CardTitle>
                <CardDescription>
                  Complete implementation with detailed comments and explanations
                </CardDescription>
              </CardHeader>
              <CardContent>
                <CodeViewer projectSlug={project.slug} />
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}