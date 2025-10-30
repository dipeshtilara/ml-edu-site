'use client'

import React, { useState, useEffect } from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'

interface KMeansVizProps {
  isRunning: boolean
  step: number
}

export function KMeansViz({ isRunning, step }: KMeansVizProps) {
  const [customers, setCustomers] = useState<any[]>([])
  const [centroids, setCentroids] = useState<any[]>([])
  const [clusters, setClusters] = useState<any[]>([])
  const [elbowData, setElbowData] = useState<any[]>([])
  const [iteration, setIteration] = useState(0)

  const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
  const clusterNames = ['🏆 Premium', '🛒 Regular', '🎯 Casual', '💰 Bargain']

  // Generate customer data
  useEffect(() => {
    if (step >= 2) {
      const customerData = Array.from({ length: 100 }, (_, i) => {
        // Create 4 natural customer segments
        const segment = Math.floor(Math.random() * 4)
        let spending, frequency
        
        switch (segment) {
          case 0: // Premium customers
            spending = 800 + Math.random() * 600
            frequency = 15 + Math.random() * 10
            break
          case 1: // Regular customers  
            spending = 300 + Math.random() * 300
            frequency = 8 + Math.random() * 8
            break
          case 2: // Casual customers
            spending = 100 + Math.random() * 200
            frequency = 2 + Math.random() * 6
            break
          default: // Bargain hunters
            spending = 50 + Math.random() * 150
            frequency = 4 + Math.random() * 8
            break
        }
        
        return {
          id: i + 1,
          spending: Math.round(spending),
          frequency: Math.round(frequency * 10) / 10,
          segment: segment,
          cluster: -1,
          name: `Customer ${i + 1}`
        }
      })
      setCustomers(customerData)
    }
  }, [step])

  // Generate elbow method data
  useEffect(() => {
    if (step >= 3) {
      const elbowPoints = [
        { k: 1, inertia: 285000, color: '#ef4444' },
        { k: 2, inertia: 189000, color: '#f59e0b' },
        { k: 3, inertia: 123000, color: '#eab308' },
        { k: 4, inertia: 89000, color: '#22c55e' }, // Optimal
        { k: 5, inertia: 72000, color: '#3b82f6' },
        { k: 6, inertia: 63000, color: '#8b5cf6' },
        { k: 7, inertia: 58000, color: '#6b7280' }
      ]
      setElbowData(elbowPoints)
    }
  }, [step])

  // Simulate K-means clustering
  useEffect(() => {
    if (step >= 5 && isRunning && customers.length > 0) {
      let currentIteration = 0
      const maxIterations = 8
      
      const runClustering = () => {
        if (currentIteration < maxIterations) {
          // Initialize or update centroids
          if (currentIteration === 0) {
            const initialCentroids = [
              { x: 200, y: 5, cluster: 0 },
              { x: 500, y: 12, cluster: 1 },
              { x: 800, y: 18, cluster: 2 },
              { x: 1200, y: 25, cluster: 3 }
            ]
            setCentroids(initialCentroids)
          }
          
          // Assign customers to nearest centroid
          const updatedCustomers = customers.map(customer => {
            let minDistance = Infinity
            let assignedCluster = 0
            
            centroids.forEach((centroid, idx) => {
              const distance = Math.sqrt(
                Math.pow(customer.spending - centroid.x, 2) + 
                Math.pow(customer.frequency - centroid.y, 2)
              )
              if (distance < minDistance) {
                minDistance = distance
                assignedCluster = idx
              }
            })
            
            return { ...customer, cluster: assignedCluster }
          })
          
          setCustomers(updatedCustomers)
          
          // Update centroids based on assignments
          const newCentroids = centroids.map((_, clusterIdx) => {
            const clusterCustomers = updatedCustomers.filter(c => c.cluster === clusterIdx)
            if (clusterCustomers.length > 0) {
              const avgSpending = clusterCustomers.reduce((sum, c) => sum + c.spending, 0) / clusterCustomers.length
              const avgFrequency = clusterCustomers.reduce((sum, c) => sum + c.frequency, 0) / clusterCustomers.length
              return { x: avgSpending, y: avgFrequency, cluster: clusterIdx }
            }
            return centroids[clusterIdx]
          })
          
          setCentroids(newCentroids)
          setIteration(currentIteration + 1)
          currentIteration++
          
          setTimeout(runClustering, 1000)
        }
      }
      
      runClustering()
    }
  }, [step, isRunning, customers.length])

  // Generate cluster analysis after clustering
  useEffect(() => {
    if (step >= 6 && customers.length > 0) {
      const clusterAnalysis = [0, 1, 2, 3].map(clusterIdx => {
        const clusterCustomers = customers.filter(c => c.cluster === clusterIdx)
        const avgSpending = clusterCustomers.reduce((sum, c) => sum + c.spending, 0) / clusterCustomers.length
        const avgFrequency = clusterCustomers.reduce((sum, c) => sum + c.frequency, 0) / clusterCustomers.length
        
        return {
          cluster: clusterIdx,
          name: clusterNames[clusterIdx],
          count: clusterCustomers.length,
          avgSpending: Math.round(avgSpending),
          avgFrequency: Math.round(avgFrequency * 10) / 10,
          percentage: (clusterCustomers.length / customers.length * 100).toFixed(1)
        }
      })
      setClusters(clusterAnalysis)
    }
  }, [step, customers])

  const getStepTitle = () => {
    switch (step) {
      case 0: return "🚀 Initializing Customer Segmentation"
      case 1: return "🛍️ Understanding Customer Behavior"
      case 2: return "📊 Loading Customer Data"
      case 3: return "🔍 Finding Optimal Number of Groups (Elbow Method)"
      case 4: return "⚙️ Setting Up K-Means Algorithm"
      case 5: return "🧠 Clustering Customers (AI Learning)"
      case 6: return "📈 Analyzing Customer Segments"
      case 7: return "🎉 Business Insights & Strategy"
      default: return "Customer Segmentation Analysis"
    }
  }

  return (
    <div className="space-y-6">
      {/* Progress Header */}
      <Card className="bg-gradient-to-r from-purple-50 to-indigo-50">
        <CardHeader>
          <CardTitle className="text-xl flex items-center">
            {getStepTitle()}
            {step === 5 && isRunning && <span className="ml-3 text-sm">Iteration: {iteration}/8</span>}
          </CardTitle>
          <Progress value={(step / 7) * 100} className="w-full" />
        </CardHeader>
      </Card>

      {/* Step 1: Problem Understanding */}
      {step >= 1 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">🛍️ What We're Trying to Discover</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-purple-50 p-6 rounded-lg">
              <div className="text-center">
                <div className="text-4xl mb-4">👥 ➜ 🎯 ➜ 💼</div>
                <p className="text-lg font-semibold text-purple-800">
                  Group Similar Customers = Better Business Strategy!
                </p>
                <p className="text-purple-600 mt-2">
                  Let's find hidden patterns in customer behavior to create targeted marketing!
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 2: Customer Data Visualization */}
      {step >= 2 && customers.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>💰 Customer Spending Patterns</CardTitle>
              <CardDescription>
                Each dot is a customer - can you see natural groups?
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="spending" 
                    name="Monthly Spending"
                    label={{ value: 'Monthly Spending ($)', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis 
                    dataKey="frequency" 
                    name="Visit Frequency"
                    label={{ value: 'Visits per Month', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    formatter={(value, name) => [
                      `${value}${name === 'spending' ? '$' : ' visits'}`, 
                      name === 'spending' ? 'Monthly Spending' : 'Visit Frequency'
                    ]}
                    labelFormatter={(props) => `Customer Data`}
                  />
                  <Scatter 
                    data={customers} 
                    dataKey="frequency" 
                    fill="#8884d8" 
                    stroke="#6366f1" 
                    strokeWidth={2}
                  />
                </ScatterChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-600 mt-2 text-center">
                💡 Do you see clusters forming? Some customers spend more and visit often!
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>📊 Spending Distribution</CardTitle>
              <CardDescription>
                How much do our customers typically spend?
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[
                  { range: '$0-200', customers: customers.filter(c => c.spending < 200).length, color: '#ef4444' },
                  { range: '$200-500', customers: customers.filter(c => c.spending >= 200 && c.spending < 500).length, color: '#f59e0b' },
                  { range: '$500-800', customers: customers.filter(c => c.spending >= 500 && c.spending < 800).length, color: '#10b981' },
                  { range: '$800+', customers: customers.filter(c => c.spending >= 800).length, color: '#3b82f6' }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value} customers`, 'Count']} />
                  <Bar dataKey="customers" fill="#8884d8" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-600 mt-2 text-center">
                💳 We have customers across all spending levels!
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Step 3: Elbow Method */}
      {step >= 3 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              🔍 Finding the Perfect Number of Groups
              <span className="ml-3 text-sm bg-yellow-100 text-yellow-800 px-2 py-1 rounded">Elbow Method</span>
            </CardTitle>
            <CardDescription>
              How many customer groups should we create? Let's find out!
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={elbowData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="k" label={{ value: 'Number of Groups (K)', position: 'insideBottom', offset: -10 }} />
                    <YAxis label={{ value: 'Error', angle: -90, position: 'insideLeft' }} />
                    <Tooltip formatter={(value) => [`${value}`, 'Clustering Error']} />
                    <Line 
                      type="monotone" 
                      dataKey="inertia" 
                      stroke="#8884d8" 
                      strokeWidth={4}
                      dot={{ r: 8, strokeWidth: 2 }}
                      activeDot={{ r: 10, fill: '#22c55e' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
                <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200">
                  <div className="flex items-center">
                    <div className="text-2xl mr-3">💡</div>
                    <div>
                      <div className="font-semibold text-green-800">The "Elbow" at K=4!</div>
                      <div className="text-sm text-green-600">
                        The curve bends most at 4 groups - that's our sweet spot!
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                  <div className="text-lg font-bold text-blue-800">Why K=4?</div>
                  <ul className="text-sm text-blue-600 mt-2 space-y-1">
                    <li>✓ Biggest drop in error</li>
                    <li>✓ Not too many groups</li>
                    <li>✓ Not too few groups</li>
                    <li>✓ Perfect for business!</li>
                  </ul>
                </div>
                
                <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                  <div className="text-sm text-yellow-800">
                    <strong>What's the Elbow Method?</strong><br/>
                    We test different numbers of groups and pick the "elbow" point where adding more groups doesn't help much!
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 5: Clustering Animation */}
      {step >= 5 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              🧠 Watch AI Learn Customer Groups
              {isRunning && <div className="ml-3 animate-pulse text-blue-600">⚡ Learning...</div>}
            </CardTitle>
            <CardDescription>
              The algorithm finds the center of each group and assigns customers!
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="spending" 
                  domain={[0, 1400]}
                  label={{ value: 'Monthly Spending ($)', position: 'insideBottom', offset: -10 }}
                />
                <YAxis 
                  dataKey="frequency" 
                  domain={[0, 30]}
                  label={{ value: 'Visits per Month', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  formatter={(value, name, props) => {
                    const customer = props.payload
                    return [
                      `${customer.name}: $${customer.spending}, ${customer.frequency} visits`,
                      clusterNames[customer.cluster] || 'Unassigned'
                    ]
                  }}
                />
                
                {/* Render customers colored by cluster */}
                {[0, 1, 2, 3].map(clusterIdx => (
                  <Scatter
                    key={clusterIdx}
                    data={customers.filter(c => c.cluster === clusterIdx)}
                    dataKey="frequency"
                    fill={colors[clusterIdx]}
                    stroke={colors[clusterIdx]}
                    strokeWidth={2}
                    name={clusterNames[clusterIdx]}
                  />
                ))}
                
                {/* Render centroids */}
                <Scatter
                  data={centroids}
                  dataKey="y"
                  fill="#000000"
                  stroke="#ffffff"
                  strokeWidth={3}
                  shape="star"
                  name="Group Centers"
                />
              </ScatterChart>
            </ResponsiveContainer>
            
            <div className="mt-4 grid grid-cols-2 lg:grid-cols-4 gap-4">
              {clusterNames.map((name, idx) => (
                <div key={idx} className="flex items-center justify-center p-2 bg-white rounded border" style={{ borderColor: colors[idx] }}>
                  <div className="w-4 h-4 rounded-full mr-2" style={{ backgroundColor: colors[idx] }}></div>
                  <span className="text-sm font-medium">{name}</span>
                </div>
              ))}
            </div>
            
            {isRunning && (
              <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                <div className="text-sm text-blue-800">
                  ⭐ Black stars = Group centers | 🔄 Iteration {iteration}/8 | 
                  The AI moves group centers and reassigns customers until groups stabilize!
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Step 6: Cluster Analysis */}
      {step >= 6 && clusters.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>📊 Customer Segment Sizes</CardTitle>
              <CardDescription>
                How many customers in each group?
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={clusters}
                    dataKey="count"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    label={({ name, percentage }) => `${name}: ${percentage}%`}
                  >
                    {clusters.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={colors[index]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value, name) => [`${value} customers`, name]} />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>💰 Average Spending by Group</CardTitle>
              <CardDescription>
                Which groups spend the most?
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={clusters}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`$${value}`, 'Average Spending']} />
                  <Bar dataKey="avgSpending" radius={[4, 4, 0, 0]}>
                    {clusters.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={colors[index]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Step 7: Business Insights */}
      {step >= 7 && (
        <Card className="bg-gradient-to-r from-green-50 to-emerald-50 border-green-200">
          <CardHeader>
            <CardTitle className="text-xl text-green-800">🎯 Business Strategy Insights</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              {clusters.map((cluster, idx) => (
                <div key={idx} className="p-4 bg-white rounded-lg border-2" style={{ borderColor: colors[idx] }}>
                  <div className="text-center">
                    <div className="text-2xl mb-2">{cluster.name.split(' ')[0]}</div>
                    <div className="font-semibold" style={{ color: colors[idx] }}>
                      {cluster.name.split(' ')[1]} Customers
                    </div>
                    <div className="text-sm text-gray-600 mt-2">
                      {cluster.count} customers ({cluster.percentage}%)
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      Avg: ${cluster.avgSpending}/month
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="p-4 bg-white rounded-lg border border-green-200">
                <h4 className="font-semibold text-green-800 mb-3">🎯 Marketing Strategies:</h4>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start">
                    <div className="w-4 h-4 rounded-full mt-1 mr-3" style={{ backgroundColor: colors[0] }}></div>
                    <div>
                      <strong>Premium Customers:</strong> VIP treatment, exclusive products, personal shopping
                    </div>
                  </div>
                  <div className="flex items-start">
                    <div className="w-4 h-4 rounded-full mt-1 mr-3" style={{ backgroundColor: colors[1] }}></div>
                    <div>
                      <strong>Regular Customers:</strong> Loyalty programs, personalized recommendations
                    </div>
                  </div>
                  <div className="flex items-start">
                    <div className="w-4 h-4 rounded-full mt-1 mr-3" style={{ backgroundColor: colors[2] }}></div>
                    <div>
                      <strong>Casual Customers:</strong> Engagement campaigns, special offers to increase visits
                    </div>
                  </div>
                  <div className="flex items-start">
                    <div className="w-4 h-4 rounded-full mt-1 mr-3" style={{ backgroundColor: colors[3] }}></div>
                    <div>
                      <strong>Bargain Customers:</strong> Discount programs, bulk deals, clearance alerts
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="p-4 bg-white rounded-lg border border-green-200">
                <h4 className="font-semibold text-green-800 mb-3">🧠 What K-Means Teaches Us:</h4>
                <ul className="space-y-2 text-sm text-green-700">
                  <li className="flex items-start">
                    <span className="text-green-500 mr-2">✓</span>
                    <span>Unsupervised learning finds hidden patterns without labels</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-green-500 mr-2">✓</span>
                    <span>Customers naturally group into similar behavior patterns</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-green-500 mr-2">✓</span>
                    <span>Data-driven segmentation leads to better business decisions</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-green-500 mr-2">✓</span>
                    <span>AI can automatically discover customer preferences</span>
                  </li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}