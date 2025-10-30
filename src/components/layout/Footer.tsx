import React from 'react'
import { Brain, Github, Mail } from 'lucide-react'

export function Footer() {
  return (
    <footer className="border-t bg-muted/50">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <Brain className="h-6 w-6 text-primary" />
              <span className="text-lg font-semibold">AI Learning Hub</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Comprehensive AI and Machine Learning projects designed for CBSE Class 12 students.
            </p>
          </div>
          
          <div className="space-y-3">
            <h4 className="text-sm font-semibold">Quick Links</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li><a href="/" className="hover:text-foreground transition-colors">Home</a></li>
              <li><a href="/projects" className="hover:text-foreground transition-colors">All Projects</a></li>
              <li><a href="/cbse-mapping" className="hover:text-foreground transition-colors">CBSE Syllabus</a></li>
              <li><a href="/about" className="hover:text-foreground transition-colors">About</a></li>
            </ul>
          </div>
          
          <div className="space-y-3">
            <h4 className="text-sm font-semibold">Categories</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li><span className="hover:text-foreground transition-colors">Supervised Learning</span></li>
              <li><span className="hover:text-foreground transition-colors">Unsupervised Learning</span></li>
              <li><span className="hover:text-foreground transition-colors">Deep Learning</span></li>
              <li><span className="hover:text-foreground transition-colors">NLP & Computer Vision</span></li>
            </ul>
          </div>
          
          <div className="space-y-3">
            <h4 className="text-sm font-semibold">Connect</h4>
            <div className="flex space-x-3">
              <Github className="h-5 w-5 text-muted-foreground hover:text-foreground cursor-pointer transition-colors" />
              <Mail className="h-5 w-5 text-muted-foreground hover:text-foreground cursor-pointer transition-colors" />
            </div>
            <p className="text-xs text-muted-foreground">
              Built for educational purposes with ❤️
            </p>
          </div>
        </div>
        
        <div className="border-t mt-8 pt-8 text-center text-sm text-muted-foreground">
          <p>&copy; 2024 AI Learning Hub. All rights reserved. Designed for CBSE Class 12 Students.</p>
        </div>
      </div>
    </footer>
  )
}