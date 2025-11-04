'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

interface NLPClassificationVizProps {
  currentStep: number;
  progress: number;
}

const NLPClassificationViz: React.FC<NLPClassificationVizProps> = ({ currentStep, progress }) => {
  const [processedDocs, setProcessedDocs] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setProcessedDocs((prev) => Math.min(prev + 1, 30));
    }, 100);
    return () => clearInterval(timer);
  }, []);

  // Sample classifications
  const classifications = [
    { category: 'Technology', count: 10, color: '#3b82f6', accuracy: 90 },
    { category: 'Sports', count: 10, color: '#10b981', accuracy: 95 },
    { category: 'Business', count: 10, color: '#f59e0b', accuracy: 85 },
  ];

  // TF-IDF top terms
  const topTerms = [
    { term: 'technology', tfidf: 0.42, category: 'Technology' },
    { term: 'player', tfidf: 0.38, category: 'Sports' },
    { term: 'company', tfidf: 0.35, category: 'Business' },
    { term: 'software', tfidf: 0.31, category: 'Technology' },
    { term: 'team', tfidf: 0.29, category: 'Sports' },
    { term: 'market', tfidf: 0.27, category: 'Business' },
  ];

  // Confusion matrix data
  const confusionMatrix = [
    { predicted: 'Tech', tech: 9, sports: 0, business: 1 },
    { predicted: 'Sports', tech: 1, sports: 9, business: 0 },
    { predicted: 'Business', tech: 0, sports: 1, business: 9 },
  ];

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="p-6 bg-gradient-to-r from-blue-50 to-purple-50">
        <h3 className="text-xl font-bold mb-2">NLP: Text Classification System</h3>
        <p className="text-gray-600">
          Multi-class text classification using TF-IDF and Naive Bayes
        </p>
        <div className="mt-4 grid grid-cols-4 gap-4">
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Documents</div>
            <div className="text-2xl font-bold text-blue-600">{processedDocs}</div>
          </div>
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Vocabulary</div>
            <div className="text-2xl font-bold text-purple-600">{Math.min(processedDocs * 15, 450)}</div>
          </div>
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Categories</div>
            <div className="text-2xl font-bold text-pink-600">3</div>
          </div>
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Accuracy</div>
            <div className="text-2xl font-bold text-green-600">90%</div>
          </div>
        </div>
      </Card>

      {/* Text Preprocessing Pipeline */}
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Text Preprocessing Pipeline</h4>
        <div className="space-y-3">
          <div className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
            <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">
              1
            </div>
            <div className="flex-1">
              <div className="font-semibold">Tokenization</div>
              <div className="text-sm text-gray-600">Split text into words</div>
            </div>
            <div className="text-sm text-gray-500">450 tokens</div>
          </div>
          <div className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
            <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">
              2
            </div>
            <div className="flex-1">
              <div className="font-semibold">Stop Word Removal</div>
              <div className="text-sm text-gray-600">Remove common words (the, is, and...)</div>
            </div>
            <div className="text-sm text-gray-500">→ 280 tokens</div>
          </div>
          <div className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
            <div className="w-8 h-8 bg-pink-500 text-white rounded-full flex items-center justify-center font-bold">
              3
            </div>
            <div className="flex-1">
              <div className="font-semibold">TF-IDF Vectorization</div>
              <div className="text-sm text-gray-600">Convert to numerical features</div>
            </div>
            <div className="text-sm text-gray-500">280D vectors</div>
          </div>
        </div>
      </Card>

      {/* Document Distribution */}
      <div className="grid grid-cols-2 gap-6">
        <Card className="p-6">
          <h4 className="text-lg font-semibold mb-4">Category Distribution</h4>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={classifications.slice(0, Math.max(1, Math.floor(processedDocs / 10)))}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ category, count }) => `${category}: ${count}`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="count"
              >
                {classifications.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card className="p-6">
          <h4 className="text-lg font-semibold mb-4">Per-Category Accuracy</h4>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={classifications}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="accuracy" fill="#10b981" name="Accuracy %" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Top TF-IDF Terms */}
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Top TF-IDF Terms by Category</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={topTerms.slice(0, Math.max(1, Math.floor(processedDocs / 5)))} layout="horizontal">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 0.5]} />
            <YAxis type="category" dataKey="term" width={100} />
            <Tooltip />
            <Legend />
            <Bar dataKey="tfidf" fill="#8b5cf6" name="TF-IDF Score" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Sample Prediction */}
      {processedDocs > 20 && (
        <Card className="p-6 bg-gradient-to-r from-green-50 to-blue-50">
          <h4 className="text-lg font-semibold mb-4">Sample Classification</h4>
          <div className="space-y-4">
            <div className="bg-white p-4 rounded-lg">
              <div className="text-sm font-semibold text-gray-600 mb-2">Input Text:</div>
              <div className="text-gray-800">
                &quot;New smartphone features advanced AI processor and 5G connectivity&quot;
              </div>
            </div>
            <div className="bg-white p-4 rounded-lg">
              <div className="text-sm font-semibold text-gray-600 mb-3">Prediction Probabilities:</div>
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Technology</span>
                    <span className="font-semibold">85%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-blue-500 h-2 rounded-full" style={{ width: '85%' }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Sports</span>
                    <span className="font-semibold">8%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{ width: '8%' }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Business</span>
                    <span className="font-semibold">7%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-orange-500 h-2 rounded-full" style={{ width: '7%' }}></div>
                  </div>
                </div>
              </div>
              <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                <div className="flex items-center gap-2">
                  <span className="text-2xl">✓</span>
                  <div>
                    <div className="font-semibold text-blue-800">Predicted: Technology</div>
                    <div className="text-sm text-blue-600">Confidence: 85%</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Model Insights */}
      <Card className="p-6 bg-gradient-to-r from-yellow-50 to-orange-50">
        <h4 className="text-lg font-semibold mb-3">Key NLP Techniques</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl mb-2">📝</div>
            <div className="font-semibold mb-1">TF-IDF</div>
            <div className="text-sm text-gray-600">
              Term Frequency-Inverse Document Frequency captures word importance
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl mb-2">🎯</div>
            <div className="font-semibold mb-1">Naive Bayes</div>
            <div className="text-sm text-gray-600">
              Probabilistic classifier with independence assumption
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl mb-2">🔤</div>
            <div className="font-semibold mb-1">Tokenization</div>
            <div className="text-sm text-gray-600">
              Breaking text into meaningful units (words)
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl mb-2">🎓</div>
            <div className="font-semibold mb-1">Laplace Smoothing</div>
            <div className="text-sm text-gray-600">
              Handles zero probabilities for unseen words
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default NLPClassificationViz;
