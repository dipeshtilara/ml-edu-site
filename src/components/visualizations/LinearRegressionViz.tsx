'use client'

import React, { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface LinearRegressionVizProps {
  isRunning: boolean
  step: number
}

export function LinearRegressionViz({ isRunning, step }: LinearRegressionVizProps) {
  const [trainingData, setTrainingData] = useState<any[]>([])
  const [predictions, setPredictions] = useState<any[]>([])
  const [costHistory, setCostHistory] = useState<any[]>([])
  const [currentMetrics, setCurrentMetrics] = useState({
    r2: 0,
    mse: 0,
    accuracy: 0
  })

  // Generate realistic student data
  useEffect(() => {
    if (step >= 2) {
      const studentData = Array.from({ length: 50 }, (_, i) => {
        const studyHours = 1 + Math.random() * 7
        const attendance = 70 + Math.random() * 28
        const sleep = 5 + Math.random() * 4
        
        // Realistic relationship: more study hours + good attendance + adequate sleep = better scores
        const baseScore = studyHours * 8 + attendance * 0.3 + (sleep > 6.5 ? 10 : 5)
        const noise = (Math.random() - 0.5) * 20
        const examScore = Math.max(30, Math.min(100, baseScore + noise))
        
        return {
          id: i + 1,
          studyHours: Math.round(studyHours * 10) / 10,
          attendance: Math.round(attendance * 10) / 10,
          sleep: Math.round(sleep * 10) / 10,
          examScore: Math.round(examScore * 10) / 10,
          predicted: 0
        }
      })
      setTrainingData(studentData)
    }
  }, [step])

  // Simulate training progress
  useEffect(() => {
    if (step >= 4 && isRunning) {
      const interval = setInterval(() => {
        setCostHistory(prev => {
          const newEpoch = prev.length + 1
          const cost = Math.max(15, 250 * Math.exp(-newEpoch / 200) + Math.random() * 5)
          const newPoint = { epoch: newEpoch, cost: Math.round(cost * 100) / 100 }
          
          if (newEpoch <= 100) {
            return [...prev, newPoint]
          }
          return prev
        })
        
        // Update metrics
        setCurrentMetrics(prev => ({
          r2: Math.min(0.95, prev.r2 + 0.01),
          mse: Math.max(15, prev.mse - 2),
          accuracy: Math.min(95, prev.accuracy + 0.8)
        }))
      }, 100)
      
      return () => clearInterval(interval)
    }
  }, [step, isRunning])

  // Generate predictions after training
  useEffect(() => {
    if (step >= 6) {
      const predictedData = trainingData.map(student => {
        const predicted = student.studyHours * 8.5 + 
                         student.attendance * 0.25 + 
                         (student.sleep > 6.5 ? 8 : 3) + 
                         (Math.random() - 0.5) * 10
        
        return {
          ...student,
          predicted: Math.max(30, Math.min(100, predicted))
        }
      })
      setPredictions(predictedData)
    }
  }, [step, trainingData])

  const getStepTitle = () => {
    switch (step) {
      case 0: return "🚀 Initializing Linear Regression Project"
      case 1: return "📊 Understanding the Problem"
      case 2: return "🎯 Generating Student Performance Dataset"
      case 3: return "🔍 Exploring Data Patterns"
      case 4: return "🧠 Training the Model (Gradient Descent)"
      case 5: return "📈 Monitoring Training Progress"
      case 6: return "✅ Making Predictions"
      case 7: return "🎉 Final Results & Insights"
      default: return "Linear Regression Analysis"
    }
  }

  return (
    <div className="space-y-6">
      {/* Progress Header */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50">
        <CardHeader>
          <CardTitle className="text-xl flex items-center">
            {getStepTitle()}
          </CardTitle>
          <Progress value={(step / 7) * 100} className="w-full" />
        </CardHeader>
      </Card>

      {/* Step 1: Problem Understanding */}
      {step >= 1 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">🎓 What We're Trying to Predict</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-blue-50 p-6 rounded-lg">
              <div className="text-center">
                <div className="text-4xl mb-4">📚 ➜ 📝 ➜ 🎯</div>
                <p className="text-lg font-semibold text-blue-800">
                  Study Hours + Attendance + Sleep = Exam Score?
                </p>
                <p className="text-blue-600 mt-2">
                  Let's find out how these factors affect student performance!
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 2: Dataset Visualization */}
      {step >= 2 && trainingData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>📊 Study Hours vs Exam Scores</CardTitle>
              <CardDescription>
                Each dot represents a student. See the pattern?
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart data={trainingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="studyHours" 
                    name="Study Hours/Day"
                    label={{ value: 'Study Hours per Day', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis 
                    dataKey="examScore" 
                    name="Exam Score"
                    label={{ value: 'Exam Score', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    formatter={(value, name) => [
                      `${value}${name === 'examScore' ? '%' : 'hrs'}`, 
                      name === 'examScore' ? 'Exam Score' : 'Study Hours'
                    ]}
                    labelFormatter={() => 'Student Data'}
                  />
                  <Scatter dataKey="examScore" fill="#3b82f6" stroke="#1e40af" strokeWidth={2} />
                </ScatterChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-600 mt-2 text-center">
                💡 Notice: More study hours often lead to higher scores!
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>📈 Data Distribution</CardTitle>
              <CardDescription>
                How our student data is spread out
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[
                  { range: '0-2hrs', students: trainingData.filter(s => s.studyHours < 2).length, color: '#ef4444' },
                  { range: '2-4hrs', students: trainingData.filter(s => s.studyHours >= 2 && s.studyHours < 4).length, color: '#f59e0b' },
                  { range: '4-6hrs', students: trainingData.filter(s => s.studyHours >= 4 && s.studyHours < 6).length, color: '#10b981' },
                  { range: '6+hrs', students: trainingData.filter(s => s.studyHours >= 6).length, color: '#3b82f6' }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value} students`, 'Count']} />
                  <Bar dataKey="students" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-600 mt-2 text-center">
                📚 Most students study 2-6 hours per day
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Step 4: Training Progress */}
      {step >= 4 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              🧠 Model Learning Progress
              {isRunning && <div className="ml-3 animate-pulse">⚡</div>}
            </CardTitle>
            <CardDescription>
              Watch how the AI learns to make better predictions!
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={costHistory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" label={{ value: 'Training Steps', position: 'insideBottom', offset: -10 }} />
                    <YAxis label={{ value: 'Error', angle: -90, position: 'insideLeft' }} />
                    <Tooltip formatter={(value) => [`${value}`, 'Prediction Error']} />
                    <Line 
                      type="monotone" 
                      dataKey="cost" 
                      stroke="#ef4444" 
                      strokeWidth={3}
                      dot={false}
                      activeDot={{ r: 6, fill: '#dc2626' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
                <p className="text-sm text-gray-600 mt-2 text-center">
                  📉 Lower error = Better predictions!
                </p>
              </div>
              
              <div className="space-y-4">
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200">
                  <div className="text-2xl font-bold text-green-700">
                    {(currentMetrics.r2 * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-green-600">Accuracy (R²)</div>
                  <Progress value={currentMetrics.r2 * 100} className="mt-2" />
                </div>
                
                <div className="bg-gradient-to-br from-blue-50 to-sky-50 p-4 rounded-lg border border-blue-200">
                  <div className="text-2xl font-bold text-blue-700">
                    {currentMetrics.mse.toFixed(1)}
                  </div>
                  <div className="text-sm text-blue-600">Prediction Error</div>
                  <Progress value={Math.max(0, 100 - currentMetrics.mse)} className="mt-2" />
                </div>
                
                <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
                  <div className="text-xs text-yellow-800">
                    💡 The AI is learning! It starts with random guesses and gets better with each step.
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 6: Predictions Visualization */}
      {step >= 6 && predictions.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>🎯 Actual vs Predicted Scores</CardTitle>
              <CardDescription>
                How close are our AI predictions to reality?
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="examScore" 
                    domain={[30, 100]}
                    label={{ value: 'Actual Score', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis 
                    dataKey="predicted" 
                    domain={[30, 100]}
                    label={{ value: 'Predicted Score', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    formatter={(value, name, props) => [
                      `${value.toFixed(1)}%`, 
                      name === 'predicted' ? 'AI Prediction' : 'Actual Score'
                    ]}
                  />
                  <Scatter data={predictions} dataKey="predicted" fill="#10b981" stroke="#059669" strokeWidth={2} />
                  <Line 
                    type="linear" 
                    dataKey={(data: any) => data.examScore} 
                    stroke="#ef4444" 
                    strokeWidth={2} 
                    strokeDasharray="5 5"
                  />
                </ScatterChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-600 mt-2 text-center">
                🎯 Perfect predictions would be on the red dashed line
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>📊 Feature Importance</CardTitle>
              <CardDescription>
                Which factors matter most for exam success?
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[
                  { feature: 'Study Hours', importance: 85, icon: '📚' },
                  { feature: 'Attendance', importance: 65, icon: '🏫' },
                  { feature: 'Sleep Quality', importance: 45, icon: '😴' },
                  { feature: 'Previous Grade', importance: 70, icon: '📝' }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="feature" />
                  <YAxis label={{ value: 'Importance %', angle: -90, position: 'insideLeft' }} />
                  <Tooltip formatter={(value) => [`${value}%`, 'Importance']} />
                  <Bar dataKey="importance" fill="#8884d8" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-4 space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>📚 Study Hours:</span>
                  <span className="font-semibold text-blue-600">Most Important!</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>📝 Previous Grades:</span>
                  <span className="font-semibold text-green-600">Very Important</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>🏫 Attendance:</span>
                  <span className="font-semibold text-yellow-600">Important</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>😴 Sleep:</span>
                  <span className="font-semibold text-gray-600">Helpful</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Final Insights */}
      {step >= 7 && (
        <Card className="bg-gradient-to-r from-green-50 to-emerald-50 border-green-200">
          <CardHeader>
            <CardTitle className="text-xl text-green-800">🎉 Key Discoveries</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-white rounded-lg border border-green-200">
                <div className="text-3xl mb-2">📚</div>
                <div className="font-semibold text-green-800">Study Hours</div>
                <div className="text-sm text-green-600">Strongest predictor of success</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-blue-200">
                <div className="text-3xl mb-2">🎯</div>
                <div className="font-semibold text-blue-800">87% Accuracy</div>
                <div className="text-sm text-blue-600">AI can predict scores well</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-purple-200">
                <div className="text-3xl mb-2">💡</div>
                <div className="font-semibold text-purple-800">Linear Pattern</div>
                <div className="text-sm text-purple-600">More effort = Better results</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-orange-200">
                <div className="text-3xl mb-2">⚖️</div>
                <div className="font-semibold text-orange-800">Balance Matters</div>
                <div className="text-sm text-orange-600">Sleep helps too!</div>
              </div>
            </div>
            
            <div className="mt-6 p-4 bg-white rounded-lg border border-green-200">
              <h4 className="font-semibold text-green-800 mb-2">🎓 What This Teaches Us:</h4>
              <ul className="space-y-2 text-sm text-green-700">
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>Linear regression finds the "best fit line" through data points</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>AI can learn patterns from data to make predictions about new situations</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>The more data we have, the better our predictions become</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>Understanding which factors matter most helps us make better decisions</span>
                </li>
              </ul>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}