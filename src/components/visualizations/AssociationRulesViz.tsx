'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';

interface AssociationRulesVizProps {
  currentStep: number;
  progress: number;
}

const AssociationRulesViz: React.FC<AssociationRulesVizProps> = ({ currentStep, progress }) => {
  const [animationStep, setAnimationStep] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setAnimationStep((prev) => Math.min(prev + 1, 10));
    }, 300);
    return () => clearInterval(timer);
  }, []);

  // Sample association rules data
  const rulesData = [
    { rule: 'Milk → Bread', support: 0.35, confidence: 0.82, lift: 2.1 },
    { rule: 'Coffee → Sugar', support: 0.28, confidence: 0.89, lift: 2.5 },
    { rule: 'Pasta → Sauce', support: 0.25, confidence: 0.76, lift: 1.9 },
    { rule: 'Eggs → Bread', support: 0.22, confidence: 0.71, lift: 1.8 },
    { rule: 'Bread,Butter → Cheese', support: 0.18, confidence: 0.68, lift: 1.7 },
  ];

  // Frequent itemsets by size
  const itemsetsData = [
    { size: '1-item', count: 12, label: '12 items' },
    { size: '2-item', count: 28, label: '28 pairs' },
    { size: '3-item', count: 15, label: '15 triplets' },
    { size: '4-item', count: 4, label: '4 quadruplets' },
  ];

  // Lift vs Support scatter
  const liftSupportData = rulesData.map((rule, idx) => ({
    support: rule.support * 100,
    lift: rule.lift,
    name: rule.rule,
    confidence: rule.confidence * 100,
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="p-6 bg-gradient-to-r from-purple-50 to-pink-50">
        <h3 className="text-xl font-bold mb-2">Association Rules Mining</h3>
        <p className="text-gray-600">
          Discovering frequent patterns and relationships in market basket transactions
        </p>
        <div className="mt-4 grid grid-cols-3 gap-4">
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Transactions</div>
            <div className="text-2xl font-bold text-purple-600">100</div>
          </div>
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Rules Found</div>
            <div className="text-2xl font-bold text-pink-600">{Math.min(animationStep * 5, 50)}</div>
          </div>
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Avg Confidence</div>
            <div className="text-2xl font-bold text-indigo-600">77%</div>
          </div>
        </div>
      </Card>

      {/* Frequent Itemsets */}
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Frequent Itemsets by Size</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={itemsetsData.slice(0, Math.max(1, Math.floor(animationStep / 2)))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="size" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" fill="#8b5cf6" name="Number of Itemsets" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Top Association Rules */}
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Top Association Rules</h4>
        <div className="space-y-3">
          {rulesData.slice(0, Math.max(1, animationStep - 2)).map((rule, idx) => (
            <div key={idx} className="border rounded-lg p-4 hover:bg-gray-50 transition-colors">
              <div className="flex justify-between items-center mb-2">
                <span className="font-mono text-sm font-semibold">{rule.rule}</span>
                <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">
                  Lift: {rule.lift.toFixed(2)}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="text-gray-600">Support</div>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${rule.support * 100}%` }}
                      />
                    </div>
                    <span className="font-semibold">{(rule.support * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div>
                  <div className="text-gray-600">Confidence</div>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${rule.confidence * 100}%` }}
                      />
                    </div>
                    <span className="font-semibold">{(rule.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div>
                  <div className="text-gray-600">Correlation</div>
                  <div className={`font-semibold ${rule.lift > 1 ? 'text-green-600' : 'text-red-600'}`}>
                    {rule.lift > 1 ? '↑ Positive' : '↓ Negative'}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Lift vs Support Scatter */}
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Rule Quality: Lift vs Support</h4>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="support" name="Support" unit="%" />
            <YAxis dataKey="lift" name="Lift" />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Legend />
            <Scatter 
              name="Association Rules" 
              data={liftSupportData.slice(0, Math.max(1, animationStep - 4))} 
              fill="#8b5cf6" 
            />
          </ScatterChart>
        </ResponsiveContainer>
        <div className="mt-4 text-sm text-gray-600">
          <p>• Lift &gt; 1: Items are positively correlated</p>
          <p>• Higher support: More frequent pattern</p>
          <p>• Confidence measures rule reliability</p>
        </div>
      </Card>

      {/* Insights */}
      <Card className="p-6 bg-gradient-to-r from-green-50 to-blue-50">
        <h4 className="text-lg font-semibold mb-3">Business Insights</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl mb-2">🛒</div>
            <div className="font-semibold">Product Placement</div>
            <div className="text-sm text-gray-600">Place associated items near each other</div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl mb-2">💡</div>
            <div className="font-semibold">Bundle Offers</div>
            <div className="text-sm text-gray-600">Create bundles based on rules</div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl mb-2">🎯</div>
            <div className="font-semibold">Cross-Selling</div>
            <div className="text-sm text-gray-600">Recommend complementary products</div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl mb-2">📊</div>
            <div className="font-semibold">Inventory</div>
            <div className="text-sm text-gray-600">Stock related items together</div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default AssociationRulesViz;
