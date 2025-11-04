'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';

interface GeneticAlgorithmVizProps {
  currentStep: number;
  progress: number;
}

const GeneticAlgorithmViz: React.FC<GeneticAlgorithmVizProps> = ({ currentStep, progress }) => {
  const [generation, setGeneration] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setGeneration((prev) => Math.min(prev + 1, 100));
    }, 80);
    return () => clearInterval(timer);
  }, []);

  // Evolution data
  const evolutionData = Array.from({ length: Math.min(generation + 1, 100) }, (_, i) => ({
    generation: i,
    bestFitness: Math.min(0.5 + (i / 100) * 0.4 + Math.random() * 0.05, 0.95),
    avgFitness: Math.min(0.3 + (i / 120) * 0.3 + Math.random() * 0.05, 0.7),
    worstFitness: Math.max(0.2 - (i / 200) * 0.1 + Math.random() * 0.05, 0.1),
  }));

  // Chromosome visualization
  const bestChromosome = Array.from({ length: 20 }, (_, i) => ({
    feature: i,
    selected: i === 0 || i === 3 || i === 7 || i === 12 || i === 15 || (generation > 50 && Math.random() > 0.8),
    important: i === 0 || i === 3 || i === 7 || i === 12 || i === 15,
  }));

  // Population diversity
  const diversityData = Array.from({ length: Math.min(generation + 1, 100) }, (_, i) => ({
    generation: i,
    diversity: Math.max(50 - i * 0.3 + Math.random() * 5, 10),
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="p-6 bg-gradient-to-r from-green-50 to-emerald-50">
        <h3 className="text-xl font-bold mb-2">Genetic Algorithm: Feature Selection</h3>
        <p className="text-gray-600">
          Evolving optimal feature subset using genetic algorithm principles
        </p>
        <div className="mt-4 grid grid-cols-4 gap-4">
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Generation</div>
            <div className="text-2xl font-bold text-green-600">{generation}</div>
          </div>
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Best Fitness</div>
            <div className="text-2xl font-bold text-emerald-600">
              {evolutionData.length > 0 ? evolutionData[evolutionData.length - 1].bestFitness.toFixed(3) : '0.000'}
            </div>
          </div>
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Population</div>
            <div className="text-2xl font-bold text-teal-600">50</div>
          </div>
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Features</div>
            <div className="text-2xl font-bold text-cyan-600">
              {bestChromosome.filter(f => f.selected).length}/20
            </div>
          </div>
        </div>
      </Card>

      {/* Chromosome Visualization */}
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Best Chromosome (Feature Selection)</h4>
        <div className="mb-4">
          <div className="flex gap-1 flex-wrap">
            {bestChromosome.map((gene, idx) => (
              <div
                key={idx}
                className={`w-8 h-8 flex items-center justify-center text-xs font-semibold rounded transition-all duration-300 ${
                  gene.selected
                    ? gene.important
                      ? 'bg-green-500 text-white'
                      : 'bg-blue-400 text-white'
                    : 'bg-gray-200 text-gray-500'
                }`}
                title={`Feature ${idx}${gene.important ? ' (Important)' : ''}`}
              >
                {gene.selected ? '1' : '0'}
              </div>
            ))}
          </div>
          <div className="mt-3 flex gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-500 rounded"></div>
              <span>Selected & Important</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-blue-400 rounded"></div>
              <span>Selected (Other)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-gray-200 rounded"></div>
              <span>Not Selected</span>
            </div>
          </div>
        </div>
      </Card>

      {/* Fitness Evolution */}
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Fitness Evolution</h4>
        <ResponsiveContainer width="100%" height={350}>
          <AreaChart data={evolutionData}>
            <defs>
              <linearGradient id="colorBest" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.1} />
              </linearGradient>
              <linearGradient id="colorAvg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="generation" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Area
              type="monotone"
              dataKey="bestFitness"
              stroke="#10b981"
              fillOpacity={1}
              fill="url(#colorBest)"
              name="Best Fitness"
            />
            <Area
              type="monotone"
              dataKey="avgFitness"
              stroke="#3b82f6"
              fillOpacity={1}
              fill="url(#colorAvg)"
              name="Avg Fitness"
            />
            <Line
              type="monotone"
              dataKey="worstFitness"
              stroke="#ef4444"
              strokeWidth={2}
              dot={false}
              name="Worst Fitness"
            />
          </AreaChart>
        </ResponsiveContainer>
      </Card>

      {/* Population Diversity */}
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Population Diversity</h4>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={diversityData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="generation" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="diversity"
              stroke="#8b5cf6"
              strokeWidth={2}
              dot={false}
              name="Genetic Diversity"
            />
          </LineChart>
        </ResponsiveContainer>
        <p className="mt-4 text-sm text-gray-600">
          Diversity decreases as population converges to optimal solution
        </p>
      </Card>

      {/* Genetic Operators */}
      <Card className="p-6 bg-gradient-to-r from-yellow-50 to-orange-50">
        <h4 className="text-lg font-semibold mb-4">Genetic Algorithm Operations</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl mb-2">🧬</div>
            <div className="font-semibold mb-1">Selection</div>
            <div className="text-sm text-gray-600">
              Tournament selection picks best individuals for reproduction
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl mb-2">✂️</div>
            <div className="font-semibold mb-1">Crossover</div>
            <div className="text-sm text-gray-600">
              Single-point crossover combines parent chromosomes (70% rate)
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl mb-2">⚡</div>
            <div className="font-semibold mb-1">Mutation</div>
            <div className="text-sm text-gray-600">
              Bit-flip mutation introduces variation (5% rate)
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl mb-2">👑</div>
            <div className="font-semibold mb-1">Elitism</div>
            <div className="text-sm text-gray-600">
              Top 2 individuals preserved each generation
            </div>
          </div>
        </div>
      </Card>

      {/* Results */}
      {generation >= 90 && (
        <Card className="p-6 bg-green-50">
          <h4 className="text-lg font-semibold mb-3 text-green-800">✓ Evolution Complete!</h4>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="font-semibold mb-2">Selected Features:</div>
              <div className="text-sm text-gray-700">
                {bestChromosome
                  .filter(g => g.selected)
                  .map(g => g.feature)
                  .join(', ')}
              </div>
            </div>
            <div>
              <div className="font-semibold mb-2">Important Features Found:</div>
              <div className="text-sm text-gray-700">
                {bestChromosome
                  .filter(g => g.selected && g.important)
                  .map(g => g.feature)
                  .join(', ')}
              </div>
            </div>
          </div>
          <div className="mt-4 text-sm text-gray-600">
            <p>• Evolved optimal feature subset in {generation} generations</p>
            <p>• Balanced accuracy vs simplicity using fitness function</p>
            <p>• Successfully identified critical features</p>
          </div>
        </Card>
      )}
    </div>
  );
};

export default GeneticAlgorithmViz;
