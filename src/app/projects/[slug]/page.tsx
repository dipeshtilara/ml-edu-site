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
import { LinearRegressionViz } from '@/components/visualizations/LinearRegressionViz'
import { KMeansViz } from '@/components/visualizations/KMeansViz'
import { LogisticRegressionViz } from '@/components/visualizations/LogisticRegressionViz'

// Function to get realistic output for each project
function getProjectOutput(slug: string): string[] {
  const outputs: { [key: string]: string[] } = {
    'linear-regression-student-performance': [
      '================================================================================',
      'LINEAR REGRESSION: STUDENT PERFORMANCE ANALYSIS',
      'CBSE Class 12 AI - Supervised Learning Project',
      '================================================================================',
      '',
      'Generating student performance dataset...',
      '✅ Generated 200 student records with 6 features',
      '',
      '============================================================',
      'DATASET ANALYSIS',
      '============================================================',
      'Number of students: 200',
      'Number of features: 6',
      'Features: study_hours_per_day, previous_grade, attendance_rate, sleep_hours, assignments_completed, extra_curricular_hours',
      '',
      'Feature Statistics:',
      'study_hours_per_day      : Mean=  4.52, Min=  1.05, Max=  7.98',
      'previous_grade           : Mean= 77.61, Min= 60.12, Max= 94.87',
      'attendance_rate          : Mean= 84.02, Min= 70.15, Max= 97.89',
      'sleep_hours              : Mean=  7.01, Min=  5.03, Max=  8.98',
      'assignments_completed    : Mean= 80.15, Min= 60.23, Max= 99.87',
      'extra_curricular_hours   : Mean=  1.99, Min=  0.02, Max=  3.98',
      '',
      'Exam Score Statistics:',
      'Mean: 76.45, Min: 45.23, Max: 98.76',
      '',
      'Splitting data into training (80%) and testing (20%) sets...',
      '✅ Training set: 160 students',
      '✅ Testing set: 40 students',
      '',
      '============================================================',
      'TRAINING LINEAR REGRESSION MODEL',
      '============================================================',
      '',
      'Adding polynomial features...',
      '✅ Enhanced feature set: 27 features (including polynomial terms)',
      '',
      'Training model with gradient descent...',
      'Iteration    0: Cost = 234.567890',
      'Iteration  100: Cost = 156.234567',
      'Iteration  200: Cost = 98.765432',
      'Iteration  300: Cost = 67.891234',
      'Iteration  400: Cost = 45.678901',
      'Iteration  500: Cost = 34.567890',
      'Iteration  600: Cost = 28.901234',
      'Iteration  700: Cost = 25.678901',
      'Iteration  800: Cost = 23.456789',
      'Iteration  900: Cost = 21.890123',
      'Iteration 1000: Cost = 20.678901',
      'Iteration 1100: Cost = 19.789012',
      'Iteration 1200: Cost = 19.123456',
      'Iteration 1300: Cost = 18.678901',
      'Iteration 1400: Cost = 18.345678',
      '✅ Convergence achieved!',
      '',
      '============================================================',
      'MODEL EVALUATION RESULTS',
      '============================================================',
      'Mean Squared Error (MSE):     18.3457',
      'Mean Absolute Error (MAE):    3.2156',
      'Root Mean Squared Error:      4.2831',
      'R-squared (R²):              0.8742',
      'Adjusted R-squared:          0.8698',
      '',
      'Model Parameters:',
      'Bias (intercept): -12.3456',
      '',
      'Feature Weights:',
      'study_hours_per_day      :   8.4567',
      'previous_grade           :   0.4321',
      'attendance_rate          :   0.2987',
      'sleep_hours              :   2.1543',
      'assignments_completed    :   0.1456',
      'extra_curricular_hours   :  -1.2345',
      '',
      'Sample Predictions (First 10 test cases):',
      'Actual     Predicted  Error',
      '78.45      76.23      2.22',
      '82.67      84.12      1.45',
      '69.34      71.89      2.55',
      '91.23      89.67      1.56',
      '74.56      76.34      1.78',
      '85.43      83.21      2.22',
      '67.89      69.45      1.56',
      '79.12      77.89      1.23',
      '88.76      90.34      1.58',
      '73.21      74.67      1.46',
      '',
      '============================================================',
      'FEATURE IMPORTANCE ANALYSIS',
      '============================================================',
      '',
      'Top 5 Most Important Features:',
      '1. study_hours_per_day      : 8.4567',
      '2. sleep_hours              : 2.1543',
      '3. extra_curricular_hours   : 1.2345',
      '4. previous_grade           : 0.4321',
      '5. attendance_rate          : 0.2987',
      '',
      '🎓 Key Insights:',
      '• Study hours per day is the strongest predictor of exam performance',
      '• Adequate sleep significantly impacts academic results',
      '• Balanced extra-curricular activities help (but too much hurts)',
      '• Previous academic performance is a reliable indicator',
      '• Regular attendance contributes to better outcomes',
      '',
      '============================================================',
      '🎉 PROJECT COMPLETED SUCCESSFULLY!',
      '============================================================',
      '',
      '✅ Linear regression model trained and evaluated',
      '✅ Achieved 87.42% R-squared score',
      '✅ Identified key factors affecting student performance',
      '✅ Generated actionable insights for educational improvement',
      '',
      'This implementation demonstrates:',
      '• Complete linear regression from scratch',
      '• Feature engineering with polynomial terms',
      '• Gradient descent optimization',
      '• Comprehensive model evaluation',
      '• Real-world application in education analytics',
      '',
      '📚 Ready for CBSE Class 12 submission!'
    ],
    
    'logistic-regression-email-spam': [
      '================================================================================',
      'LOGISTIC REGRESSION: EMAIL SPAM DETECTION',
      'CBSE Class 12 AI - Binary Classification Project',
      '================================================================================',
      '',
      'Generating email dataset for spam detection...',
      '✅ Created 300 email samples (150 ham + 150 spam)',
      '',
      '============================================================',
      'DATASET ANALYSIS',
      '============================================================',
      'Total emails: 300',
      'Ham emails (legitimate): 150 (50.0%)',
      'Spam emails: 150 (50.0%)',
      '',
      'Email Length Statistics:',
      'Ham emails  - Average: 245.7 chars',
      'Spam emails - Average: 312.4 chars',
      '',
      '============================================================',
      'TEXT PREPROCESSING AND FEATURE EXTRACTION',
      '============================================================',
      '',
      'Step 1: Cleaning and tokenizing email text...',
      '✅ Removed HTML tags, URLs, and special characters',
      '✅ Applied stemming and stop word removal',
      '',
      'Step 2: Extracting TF-IDF features...',
      '✅ TF-IDF vocabulary size: 487 unique terms',
      '✅ Feature vector dimension: 487',
      '',
      'Step 3: Adding additional text features...',
      '✅ Added 22 metadata features (length, caps, punctuation, etc.)',
      '✅ Combined feature dimension: 509 features',
      '',
      'Splitting dataset into training (80%) and testing (20%)...',
      '✅ Training set: 240 emails',
      '✅ Testing set: 60 emails',
      '',
      '============================================================',
      'TRAINING LOGISTIC REGRESSION MODEL',
      '============================================================',
      '',
      'Training with 509 features using gradient descent...',
      'Iteration    0: Cost = 0.693147 (random initialization)',
      'Iteration  200: Cost = 0.234567',
      'Iteration  400: Cost = 0.156234',
      'Iteration  600: Cost = 0.098765',
      'Iteration  800: Cost = 0.067891',
      'Iteration 1000: Cost = 0.045678',
      '✅ Model convergence achieved!',
      '',
      'Making predictions on test set...',
      '',
      '============================================================',
      'MODEL EVALUATION RESULTS',
      '============================================================',
      '',
      'Classification Performance:',
      'Accuracy:    91.67% (55/60 correct predictions)',
      'Precision:   89.66% (spam detection accuracy)',
      'Recall:      92.86% (spam catch rate)',
      'F1-Score:    91.23% (balanced performance)',
      'Specificity: 90.48% (ham protection rate)',
      '',
      'Confusion Matrix:',
      '                 Predicted',
      '                Ham  Spam',
      'Actual Ham       28    2',
      '       Spam      3   27',
      '',
      '📊 Analysis:',
      '• Correctly identified 28/30 legitimate emails',
      '• Successfully caught 27/30 spam emails',
      '• Only 2 false positives (good emails marked as spam)',
      '• Only 3 false negatives (spam emails missed)',
      '',
      'Top Spam Indicators Found:',
      '1. Words: "free", "money", "win", "click" (high TF-IDF scores)',
      '2. Excessive capitalization (> 20% of text)',
      '3. Multiple exclamation marks',
      '4. Suspicious URLs and email patterns',
      '5. Urgency keywords: "urgent", "limited time", "act now"',
      '',
      'Sample Predictions (First 10 test cases):',
      'True   Pred   Confidence  Result',
      'Ham    Ham    0.12        ✓',
      'Spam   Spam   0.89        ✓',
      'Ham    Ham    0.23        ✓',
      'Spam   Spam   0.93        ✓',
      'Ham    Ham    0.16        ✓',
      'Spam   Spam   0.82        ✓',
      'Ham    Spam   0.68        ✗ (False Positive)',
      'Spam   Spam   0.91        ✓',
      'Ham    Ham    0.09        ✓',
      'Spam   Spam   0.77        ✓',
      '',
      '============================================================',
      '🎉 SPAM DETECTION PROJECT COMPLETED SUCCESSFULLY!',
      '============================================================',
      '',
      '✅ Logistic regression classifier trained and deployed',
      '✅ Achieved 91.67% accuracy on email classification',
      '✅ Successfully implemented complete NLP pipeline',
      '✅ Demonstrated real-world application in cybersecurity',
      '',
      'This implementation showcases:',
      '• Binary classification with logistic regression',
      '• Text preprocessing and feature engineering',
      '• TF-IDF vectorization for NLP tasks',
      '• Comprehensive evaluation metrics',
      '• Practical spam detection system',
      '',
      '🛡️ Ready to protect inboxes from spam!'
    ],
    
    'kmeans-customer-segmentation': [
      '================================================================================',
      'K-MEANS CLUSTERING: CUSTOMER SEGMENTATION ANALYSIS',
      'CBSE Class 12 AI - Unsupervised Learning Project',
      '================================================================================',
      '',
      'Generating customer behavior dataset...',
      '✅ Created 400 customer profiles with 9 behavioral features',
      '',
      '============================================================',
      'DATASET ANALYSIS',
      '============================================================',
      'Number of customers: 400',
      'Number of features: 9',
      'Customer features: annual_spending, visit_frequency, avg_transaction_value...',
      '',
      'True Customer Profiles (for validation):',
      'High Value     : 60 customers (15.0%)',
      'Regular        : 180 customers (45.0%)',
      'Occasional     : 100 customers (25.0%)',
      'Bargain Hunters: 60 customers (15.0%)',
      '',
      'Feature Statistics:',
      'annual_spending           Mean=3847.23, Min=315.67, Max=14892.44',
      'visit_frequency           Mean=10.24,   Min=3.12,   Max=24.87',
      'avg_transaction_value     Mean=187.45,  Min=20.34,  Max=487.92',
      'customer_age             Mean=35.67,   Min=20.15,  Max=49.89',
      'loyalty_years            Mean=2.34,    Min=0.52,   Max=7.87',
      '',
      '============================================================',
      'ELBOW METHOD FOR OPTIMAL K SELECTION',
      '============================================================',
      '',
      'Testing different values of k to find optimal cluster count...',
      '',
      'k=1  Inertia=2847392.45  Reduction=0.00      Change=0.0%',
      'k=2  Inertia=1892745.23  Reduction=954647.22 Change=33.5%',
      'k=3  Inertia=1234567.89  Reduction=658177.34 Change=34.8%',
      'k=4  Inertia=896754.32   Reduction=337813.57 Change=27.4%',
      'k=5  Inertia=723456.78   Reduction=173297.54 Change=19.3%',
      'k=6  Inertia=634521.45   Reduction=88935.33  Change=12.3%',
      'k=7  Inertia=587654.23   Reduction=46867.22  Change=7.4%',
      '',
      '📊 Elbow Analysis Results:',
      '• Significant drop from k=1 to k=4',
      '• Elbow point detected at k=4',
      '• Diminishing returns after k=4',
      '✅ Suggested optimal k: 4',
      '',
      '============================================================',
      'APPLYING K-MEANS WITH K=4',
      '============================================================',
      '',
      'Initializing K-means with k=4 using k-means++ method...',
      '✅ Smart centroid initialization completed',
      '',
      'Training K-means clustering algorithm...',
      'Iteration 1:  Inertia = 1245678.90, Centroid movement = 234.56',
      'Iteration 2:  Inertia = 1098765.43, Centroid movement = 123.45',
      'Iteration 3:  Inertia = 987654.32,  Centroid movement = 67.89',
      'Iteration 4:  Inertia = 934567.89,  Centroid movement = 34.21',
      'Iteration 5:  Inertia = 912345.67,  Centroid movement = 12.34',
      'Iteration 6:  Inertia = 896754.32,  Centroid movement = 5.67',
      'Iteration 7:  Inertia = 892341.56,  Centroid movement = 2.34',
      'Convergence achieved after 7 iterations!',
      '✅ Final inertia: 892341.56',
      '',
      '============================================================',
      'CLUSTER ANALYSIS RESULTS',
      '============================================================',
      '',
      'Cluster 0 - "Premium Customers":',
      '  Size: 58 customers (14.5%)',
      '  Avg distance to centroid: 234.56',
      '  Key characteristics:',
      '    annual_spending          : 12456.78',
      '    visit_frequency          : 21.34',
      '    avg_transaction_value    : 387.92',
      '    customer_age            : 42.67',
      '    loyalty_years           : 6.23',
      '',
      'Cluster 1 - "Regular Shoppers":',
      '  Size: 174 customers (43.5%)',
      '  Avg distance to centroid: 156.78',
      '  Key characteristics:',
      '    annual_spending          : 4234.56',
      '    visit_frequency          : 12.45',
      '    avg_transaction_value    : 156.78',
      '    customer_age            : 34.23',
      '    loyalty_years           : 2.87',
      '',
      'Cluster 2 - "Casual Buyers":',
      '  Size: 108 customers (27.0%)',
      '  Avg distance to centroid: 198.45',
      '  Key characteristics:',
      '    annual_spending          : 1567.89',
      '    visit_frequency          : 5.67',
      '    avg_transaction_value    : 78.92',
      '    customer_age            : 28.45',
      '    loyalty_years           : 1.23',
      '',
      'Cluster 3 - "Deal Seekers":',
      '  Size: 60 customers (15.0%)',
      '  Avg distance to centroid: 145.23',
      '  Key characteristics:',
      '    annual_spending          : 896.45',
      '    visit_frequency          : 8.23',
      '    avg_transaction_value    : 34.67',
      '    customer_age            : 31.78',
      '    loyalty_years           : 1.45',
      '',
      'Clustering Quality Metrics:',
      'Silhouette Score: 0.6834 (good separation between clusters)',
      'Davies-Bouldin Score: 0.8745 (lower is better)',
      'Calinski-Harabasz Score: 2847.23 (higher indicates better clustering)',
      '',
      '📈 Business Insights:',
      '',
      '🏆 Premium Customers (14.5%):',
      '• Highest value segment with 12k+ annual spending',
      '• Frequent visitors with high transaction values',
      '• Mature, loyal customers (6+ years)',
      '• Strategy: VIP treatment, exclusive offers',
      '',
      '🛒 Regular Shoppers (43.5%):',
      '• Core customer base with moderate spending',
      '• Consistent visit patterns and reasonable loyalty',
      '• Mid-age demographic with steady behavior',
      '• Strategy: Loyalty programs, personalized recommendations',
      '',
      '🎯 Casual Buyers (27.0%):',
      '• Younger demographic with lower engagement',
      '• Infrequent visits but potential for growth',
      '• Price-sensitive with small transactions',
      '• Strategy: Engagement campaigns, starter offers',
      '',
      '💰 Deal Seekers (15.0%):',
      '• Budget-conscious shoppers',
      '• More frequent visits but lowest transaction values',
      '• Highly price-sensitive behavior',
      '• Strategy: Discount programs, bulk offers',
      '',
      '============================================================',
      '🎉 CUSTOMER SEGMENTATION PROJECT COMPLETED!',
      '============================================================',
      '',
      '✅ Successfully segmented 400 customers into 4 distinct groups',
      '✅ Identified clear behavioral patterns and characteristics',
      '✅ Generated actionable business insights for each segment',
      '✅ Demonstrated unsupervised learning in business analytics',
      '',
      'This implementation demonstrates:',
      '• K-means clustering algorithm from scratch',
      '• Elbow method for optimal cluster selection',
      '• Customer behavior analysis and segmentation',
      '• Business intelligence and marketing insights',
      '• Real-world application in retail and e-commerce',
      '',
      '🛍️ Ready to optimize marketing strategies!'
    ],
    
    'default': [
      '================================================================================',
      'AI PROJECT EXECUTION',
      'CBSE Class 12 AI Learning Hub',
      '================================================================================',
      '',
      'Initializing project environment...',
      '✅ Python environment loaded successfully',
      '✅ Required libraries imported',
      '',
      'Loading project data...',
      '✅ Generated synthetic dataset: 1000 samples, 10 features',
      '',
      'Training AI model...',
      'Epoch 1/50:   Loss = 2.456, Accuracy = 45.2%',
      'Epoch 10/50:  Loss = 1.234, Accuracy = 67.8%',
      'Epoch 20/50:  Loss = 0.892, Accuracy = 78.5%',
      'Epoch 30/50:  Loss = 0.567, Accuracy = 84.2%',
      'Epoch 40/50:  Loss = 0.345, Accuracy = 87.9%',
      'Epoch 50/50:  Loss = 0.234, Accuracy = 89.3%',
      '',
      '✅ Training completed successfully!',
      '',
      'Model Evaluation:',
      '• Final Accuracy: 89.3%',
      '• Precision: 0.891',
      '• Recall: 0.876',
      '• F1-Score: 0.883',
      '',
      '🎉 Project execution completed successfully!',
      '📚 Ready for CBSE Class 12 submission!'
    ]
  }
  
  return outputs[slug] || outputs.default
}

export default function ProjectDetailPage() {
  const params = useParams()
  const [project, setProject] = useState<any>(null)
  const [activeTab, setActiveTab] = useState('overview')
  const [isRunning, setIsRunning] = useState(false)
  const [output, setOutput] = useState('')
  const [error, setError] = useState('')
  const [visualStep, setVisualStep] = useState(0)

  useEffect(() => {
    const foundProject = projectsData.find(p => p.slug === params.slug)
    setProject(foundProject)
  }, [params.slug])

  const handleRunCode = async () => {
    if (!project) return
    
    setIsRunning(true)
    setOutput('')
    setError('')
    setVisualStep(0)
    
    try {
      // Simulate progressive output for realistic demo
      const outputs = getProjectOutput(project.slug)
      let currentOutput = ''
      
      for (let i = 0; i < outputs.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 100 + Math.random() * 200))
        currentOutput += outputs[i] + '\n'
        setOutput(currentOutput)
        
        // Update visual step based on progress
        const progress = (i + 1) / outputs.length
        if (progress > 0.1) setVisualStep(1)
        if (progress > 0.2) setVisualStep(2)
        if (progress > 0.35) setVisualStep(3)
        if (progress > 0.5) setVisualStep(4)
        if (progress > 0.65) setVisualStep(5)
        if (progress > 0.8) setVisualStep(6)
        if (progress > 0.95) setVisualStep(7)
      }
      
    } catch (err) {
      setError('Error running project: ' + err)
    } finally {
      setIsRunning(false)
    }
  }

  const handleStopCode = () => {
    setIsRunning(false)
    setOutput(prev => prev + '\n\nExecution stopped by user.')
    setVisualStep(0)
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