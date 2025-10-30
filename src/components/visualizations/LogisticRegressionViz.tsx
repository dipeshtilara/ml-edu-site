'use client'

import React, { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface LogisticRegressionVizProps {
  isRunning: boolean
  step: number
}

export function LogisticRegressionViz({ isRunning, step }: LogisticRegressionVizProps) {
  const [emails, setEmails] = useState<any[]>([])
  const [confusionMatrix, setConfusionMatrix] = useState<any>(null)
  const [trainingProgress, setTrainingProgress] = useState<any[]>([])
  const [predictions, setPredictions] = useState<any[]>([])
  const [currentAccuracy, setCurrentAccuracy] = useState(0)

  // Generate email dataset
  useEffect(() => {
    if (step >= 2) {
      const emailData = Array.from({ length: 100 }, (_, i) => {
        const isSpam = Math.random() > 0.5
        
        // Spam emails tend to be longer and have more caps
        const length = isSpam ? 200 + Math.random() * 300 : 100 + Math.random() * 200
        const capsPercentage = isSpam ? 15 + Math.random() * 30 : Math.random() * 10
        const exclamations = isSpam ? Math.floor(Math.random() * 8) : Math.floor(Math.random() * 2)
        const spamWords = isSpam ? Math.floor(2 + Math.random() * 6) : Math.floor(Math.random() * 2)
        
        return {
          id: i + 1,
          type: isSpam ? 'spam' : 'ham',
          length: Math.round(length),
          capsPercentage: Math.round(capsPercentage * 10) / 10,
          exclamations,
          spamWords,
          predicted: null,
          confidence: 0,
          subject: isSpam ? 
            ['FREE MONEY!', 'URGENT: Click Now!', 'You Won $1000!', 'Limited Time Offer!!!'][Math.floor(Math.random() * 4)] :
            ['Meeting Tomorrow', 'Project Update', 'Lunch Plans', 'Weekly Report'][Math.floor(Math.random() * 4)]
        }
      })
      setEmails(emailData)
    }
  }, [step])

  // Simulate training progress
  useEffect(() => {
    if (step >= 4 && isRunning) {
      const interval = setInterval(() => {
        setTrainingProgress(prev => {
          const epoch = prev.length + 1
          const cost = Math.max(0.05, 0.693 * Math.exp(-epoch / 50) + Math.random() * 0.1)
          const accuracy = Math.min(95, 50 + (epoch / 100) * 40 + Math.random() * 5)
          
          const newPoint = { 
            epoch, 
            cost: Math.round(cost * 1000) / 1000,
            accuracy: Math.round(accuracy * 10) / 10
          }
          
          setCurrentAccuracy(accuracy)
          
          if (epoch <= 100) {
            return [...prev, newPoint]
          }
          return prev
        })
      }, 80)
      
      return () => clearInterval(interval)
    }
  }, [step, isRunning])

  // Generate predictions and confusion matrix
  useEffect(() => {
    if (step >= 6) {
      const predictedEmails = emails.map(email => {
        // Simple heuristic based on features
        let spamScore = 0
        if (email.length > 250) spamScore += 0.3
        if (email.capsPercentage > 20) spamScore += 0.4
        if (email.exclamations > 3) spamScore += 0.2
        if (email.spamWords > 3) spamScore += 0.4
        
        const confidence = Math.min(0.99, Math.max(0.01, spamScore + (Math.random() - 0.5) * 0.3))
        const predicted = confidence > 0.5 ? 'spam' : 'ham'
        
        return {
          ...email,
          predicted,
          confidence: Math.round(confidence * 100) / 100
        }
      })
      
      setPredictions(predictedEmails)
      
      // Calculate confusion matrix
      const tp = predictedEmails.filter(e => e.type === 'spam' && e.predicted === 'spam').length
      const tn = predictedEmails.filter(e => e.type === 'ham' && e.predicted === 'ham').length
      const fp = predictedEmails.filter(e => e.type === 'ham' && e.predicted === 'spam').length
      const fn = predictedEmails.filter(e => e.type === 'spam' && e.predicted === 'ham').length
      
      setConfusionMatrix({ tp, tn, fp, fn })
    }
  }, [step, emails])

  const getStepTitle = () => {
    switch (step) {
      case 0: return "🚀 Initializing Email Spam Detection"
      case 1: return "📧 Understanding Email Classification"
      case 2: return "📊 Loading Email Dataset"
      case 3: return "🔍 Feature Engineering & Text Processing"
      case 4: return "🧠 Training Logistic Regression Model"
      case 5: return "📈 Monitoring Classification Performance"
      case 6: return "🎯 Making Predictions & Confusion Matrix"
      case 7: return "🎉 Results & Anti-Spam Insights"
      default: return "Email Spam Detection"
    }
  }

  return (
    <div className="space-y-6">
      {/* Progress Header */}
      <Card className="bg-gradient-to-r from-red-50 to-orange-50">
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
            <CardTitle className="text-lg">📧 Email Classification Challenge</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-red-50 p-6 rounded-lg">
              <div className="text-center">
                <div className="text-4xl mb-4">📧 ➜ 🤖 ➜ ✅/❌</div>
                <p className="text-lg font-semibold text-red-800">
                  Can AI Tell Spam from Real Emails?
                </p>
                <p className="text-red-600 mt-2">
                  Let's build an intelligent spam filter that protects inboxes!
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 2: Email Dataset */}
      {step >= 2 && emails.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>📊 Email Types in Our Dataset</CardTitle>
              <CardDescription>
                50% spam emails vs 50% legitimate emails
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[
                  { 
                    type: '📧 Legitimate', 
                    count: emails.filter(e => e.type === 'ham').length, 
                    color: '#22c55e' 
                  },
                  { 
                    type: '⚠️ Spam', 
                    count: emails.filter(e => e.type === 'spam').length, 
                    color: '#ef4444' 
                  }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="type" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value} emails`, 'Count']} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    <Bar dataKey="count" fill="#3b82f6" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div className="p-3 bg-green-50 rounded-lg border border-green-200">
                  <div className="text-sm font-medium text-green-800">✅ Legitimate Emails</div>
                  <div className="text-xs text-green-600 mt-1">
                    "Meeting Tomorrow", "Project Update"
                  </div>
                </div>
                <div className="p-3 bg-red-50 rounded-lg border border-red-200">
                  <div className="text-sm font-medium text-red-800">⚠️ Spam Emails</div>
                  <div className="text-xs text-red-600 mt-1">
                    "FREE MONEY!", "URGENT: Click Now!"
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>🔍 Email Features Analysis</CardTitle>
              <CardDescription>
                Key differences between spam and legitimate emails
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="length" 
                    name="Email Length"
                    label={{ value: 'Email Length (characters)', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis 
                    dataKey="capsPercentage" 
                    name="Caps %"
                    label={{ value: 'Capital Letters %', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    formatter={(value, name, props) => {
                      const email = props.payload
                      return [
                        `${email.subject} - ${email.type}`,
                        `Length: ${email.length}, Caps: ${email.capsPercentage}%`
                      ]
                    }}
                  />
                  <Scatter 
                    data={emails.filter(e => e.type === 'ham')} 
                    dataKey="capsPercentage" 
                    fill="#22c55e" 
                    name="Legitimate"
                  />
                  <Scatter 
                    data={emails.filter(e => e.type === 'spam')} 
                    dataKey="capsPercentage" 
                    fill="#ef4444" 
                    name="Spam"
                  />
                </ScatterChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-600 mt-2 text-center">
                🔍 Spam emails tend to be longer with MORE CAPITAL LETTERS!
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
              🧠 AI Learning to Detect Spam
              {isRunning && <div className="ml-3 animate-pulse">⚡</div>}
            </CardTitle>
            <CardDescription>
              Watch the AI get smarter at spotting spam emails!
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={trainingProgress}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" label={{ value: 'Training Steps', position: 'insideBottom', offset: -10 }} />
                    <YAxis yAxisId="left" label={{ value: 'Error', angle: -90, position: 'insideLeft' }} />
                    <YAxis yAxisId="right" orientation="right" label={{ value: 'Accuracy %', angle: 90, position: 'insideRight' }} />
                    <Tooltip 
                      formatter={(value, name) => [
                        name === 'cost' ? value : `${value}%`, 
                        name === 'cost' ? 'Classification Error' : 'Accuracy'
                      ]}
                    />
                    <Line 
                      yAxisId="left"
                      type="monotone" 
                      dataKey="cost" 
                      stroke="#ef4444" 
                      strokeWidth={3}
                      name="cost"
                    />
                    <Line 
                      yAxisId="right"
                      type="monotone" 
                      dataKey="accuracy" 
                      stroke="#22c55e" 
                      strokeWidth={3}
                      name="accuracy"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              
              <div className="space-y-4">
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200">
                  <div className="text-2xl font-bold text-green-700">
                    {currentAccuracy.toFixed(1)}%
                  </div>
                  <div className="text-sm text-green-600">Classification Accuracy</div>
                  <Progress value={currentAccuracy} className="mt-2" />
                </div>
                
                <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
                  <div className="text-xs text-yellow-800">
                    💡 <strong>Logistic Regression Magic:</strong> The AI uses the sigmoid function to convert any input into a probability between 0% and 100%!
                  </div>
                </div>
                
                <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                  <div className="text-xs text-blue-800">
                    🎯 <strong>Learning Process:</strong> Each training step, the AI adjusts its "spam detector" to be more accurate!
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 6: Confusion Matrix */}
      {step >= 6 && confusionMatrix && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>🎯 Confusion Matrix - How Well Did We Do?</CardTitle>
              <CardDescription>
                The ultimate test: How many emails did we classify correctly?
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div></div>
                  <div className="font-semibold text-sm bg-gray-100 p-2 rounded">Predicted</div>
                  <div></div>
                  
                  <div className="font-semibold text-sm bg-gray-100 p-2 rounded">Actual</div>
                  <div className="text-xs text-green-600">📧 Legitimate</div>
                  <div className="text-xs text-red-600">⚠️ Spam</div>
                  
                  <div className="text-xs text-green-600">📧 Legit</div>
                  <div className="bg-green-100 p-4 rounded-lg border-2 border-green-500">
                    <div className="text-2xl font-bold text-green-700">{confusionMatrix.tn}</div>
                    <div className="text-xs text-green-600">✅ Correct!</div>
                  </div>
                  <div className="bg-red-100 p-4 rounded-lg border-2 border-red-300">
                    <div className="text-2xl font-bold text-red-700">{confusionMatrix.fp}</div>
                    <div className="text-xs text-red-600">❌ False Alarm</div>
                  </div>
                  
                  <div className="text-xs text-red-600">⚠️ Spam</div>
                  <div className="bg-orange-100 p-4 rounded-lg border-2 border-orange-300">
                    <div className="text-2xl font-bold text-orange-700">{confusionMatrix.fn}</div>
                    <div className="text-xs text-orange-600">❌ Missed Spam</div>
                  </div>
                  <div className="bg-green-100 p-4 rounded-lg border-2 border-green-500">
                    <div className="text-2xl font-bold text-green-700">{confusionMatrix.tp}</div>
                    <div className="text-xs text-green-600">✅ Caught Spam!</div>
                  </div>
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>✅ Correct Predictions:</span>
                    <span className="font-semibold text-green-600">
                      {confusionMatrix.tp + confusionMatrix.tn} / {confusionMatrix.tp + confusionMatrix.tn + confusionMatrix.fp + confusionMatrix.fn}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>🎯 Accuracy:</span>
                    <span className="font-semibold text-blue-600">
                      {(((confusionMatrix.tp + confusionMatrix.tn) / (confusionMatrix.tp + confusionMatrix.tn + confusionMatrix.fp + confusionMatrix.fn)) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>📈 Performance Metrics</CardTitle>
              <CardDescription>
                Understanding our spam filter's strengths
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Calculate metrics */}
                {(() => {
                  const precision = confusionMatrix.tp / (confusionMatrix.tp + confusionMatrix.fp)
                  const recall = confusionMatrix.tp / (confusionMatrix.tp + confusionMatrix.fn)
                  const f1Score = 2 * (precision * recall) / (precision + recall)
                  
                  return (
                    <>
                      <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                        <div className="flex justify-between items-center">
                          <div>
                            <div className="font-semibold text-blue-800">Precision</div>
                            <div className="text-xs text-blue-600">When we say "spam", how often are we right?</div>
                          </div>
                          <div className="text-2xl font-bold text-blue-700">{(precision * 100).toFixed(1)}%</div>
                        </div>
                        <Progress value={precision * 100} className="mt-2" />
                      </div>
                      
                      <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                        <div className="flex justify-between items-center">
                          <div>
                            <div className="font-semibold text-green-800">Recall</div>
                            <div className="text-xs text-green-600">How many spam emails do we catch?</div>
                          </div>
                          <div className="text-2xl font-bold text-green-700">{(recall * 100).toFixed(1)}%</div>
                        </div>
                        <Progress value={recall * 100} className="mt-2" />
                      </div>
                      
                      <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                        <div className="flex justify-between items-center">
                          <div>
                            <div className="font-semibold text-purple-800">F1-Score</div>
                            <div className="text-xs text-purple-600">Overall balance of precision & recall</div>
                          </div>
                          <div className="text-2xl font-bold text-purple-700">{(f1Score * 100).toFixed(1)}%</div>
                        </div>
                        <Progress value={f1Score * 100} className="mt-2" />
                      </div>
                    </>
                  )
                })()}
                
                <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
                  <div className="text-xs text-yellow-800">
                    💡 <strong>Why These Matter:</strong><br/>
                    • High Precision = Few false alarms<br/>
                    • High Recall = Catch most spam<br/>
                    • High F1 = Good balance of both!
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Final Results */}
      {step >= 7 && (
        <Card className="bg-gradient-to-r from-green-50 to-emerald-50 border-green-200">
          <CardHeader>
            <CardTitle className="text-xl text-green-800">🛡️ Spam Filter Successfully Deployed!</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <div className="text-center p-4 bg-white rounded-lg border border-green-200">
                <div className="text-3xl mb-2">🎯</div>
                <div className="font-semibold text-green-800">High Accuracy</div>
                <div className="text-sm text-green-600">AI correctly identifies most emails</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-blue-200">
                <div className="text-3xl mb-2">⚡</div>
                <div className="font-semibold text-blue-800">Fast Processing</div>
                <div className="text-sm text-blue-600">Instant spam detection</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-purple-200">
                <div className="text-3xl mb-2">🧠</div>
                <div className="font-semibold text-purple-800">Smart Learning</div>
                <div className="text-sm text-purple-600">Learns from email patterns</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg border border-orange-200">
                <div className="text-3xl mb-2">🛡️</div>
                <div className="font-semibold text-orange-800">Inbox Protection</div>
                <div className="text-sm text-orange-600">Keeps spam out effectively</div>
              </div>
            </div>
            
            <div className="p-4 bg-white rounded-lg border border-green-200">
              <h4 className="font-semibold text-green-800 mb-2">🎓 What This Teaches Us About AI:</h4>
              <ul className="space-y-2 text-sm text-green-700">
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>Binary classification helps AI make yes/no decisions</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>Feature engineering turns text into numbers AI can understand</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>Confusion matrices help us understand AI performance clearly</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span>AI can protect us from digital threats like spam and phishing</span>
                </li>
              </ul>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}