'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface QLearningVizProps {
  currentStep: number;
  progress: number;
}

const QLearningViz: React.FC<QLearningVizProps> = ({ currentStep, progress }) => {
  const [episode, setEpisode] = useState(0);
  const [agentPosition, setAgentPosition] = useState({ row: 0, col: 0 });

  useEffect(() => {
    const timer = setInterval(() => {
      setEpisode((prev) => Math.min(prev + 1, 500));
    }, 50);
    return () => clearInterval(timer);
  }, []);

  // Maze layout (0=path, 1=wall, 2=goal, 3=start)
  const maze = [
    [3, 0, 1, 0, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 2],
  ];

  // Sample learned path
  const learnedPath = [
    { row: 0, col: 0 },
    { row: 1, col: 0 },
    { row: 1, col: 1 },
    { row: 2, col: 1 },
    { row: 2, col: 2 },
    { row: 2, col: 3 },
    { row: 3, col: 3 },
    { row: 4, col: 3 },
    { row: 4, col: 4 },
  ];

  // Training progress data
  const trainingData = Array.from({ length: Math.min(episode, 500) }, (_, i) => ({
    episode: i,
    steps: Math.max(15 - Math.floor(i / 50), 9) + Math.random() * 2,
    reward: Math.min(-50 + i / 5, 100),
    epsilon: Math.max(1 - i / 500, 0.01),
  })).filter((_, i) => i % 10 === 0);

  const getCellStyle = (row: number, col: number) => {
    const cell = maze[row][col];
    const isInPath = learnedPath.some(p => p.row === row && p.col === col);
    
    if (cell === 1) return 'bg-gray-800'; // Wall
    if (cell === 2) return 'bg-green-500 text-white font-bold flex items-center justify-center'; // Goal
    if (cell === 3) return 'bg-blue-500 text-white font-bold flex items-center justify-center'; // Start
    if (isInPath && episode > 400) return 'bg-yellow-200'; // Learned path
    return 'bg-gray-100';
  };

  const getCellContent = (row: number, col: number) => {
    const cell = maze[row][col];
    if (cell === 2) return 'G';
    if (cell === 3) return 'S';
    return '';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50">
        <h3 className="text-xl font-bold mb-2">Q-Learning: Reinforcement Learning Agent</h3>
        <p className="text-gray-600">
          Training an agent to navigate maze using Q-learning algorithm
        </p>
        <div className="mt-4 grid grid-cols-4 gap-4">
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Episode</div>
            <div className="text-2xl font-bold text-blue-600">{episode}</div>
          </div>
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Q-Table Size</div>
            <div className="text-2xl font-bold text-indigo-600">{Math.min(episode * 2, 100)}</div>
          </div>
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Exploration (ε)</div>
            <div className="text-2xl font-bold text-purple-600">
              {Math.max(1 - episode / 500, 0.01).toFixed(2)}
            </div>
          </div>
          <div className="bg-white p-3 rounded shadow-sm">
            <div className="text-sm text-gray-600">Success Rate</div>
            <div className="text-2xl font-bold text-green-600">
              {Math.min(episode / 5, 100).toFixed(0)}%
            </div>
          </div>
        </div>
      </Card>

      {/* Maze Visualization */}
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Maze Environment</h4>
        <div className="flex justify-center mb-4">
          <div className="inline-block border-2 border-gray-300 rounded-lg overflow-hidden">
            {maze.map((row, rowIdx) => (
              <div key={rowIdx} className="flex">
                {row.map((cell, colIdx) => (
                  <div
                    key={`${rowIdx}-${colIdx}`}
                    className={`w-16 h-16 border border-gray-200 ${getCellStyle(rowIdx, colIdx)} transition-colors duration-300`}
                  >
                    <div className="w-full h-full flex items-center justify-center text-xl">
                      {getCellContent(rowIdx, colIdx)}
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
        <div className="flex justify-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-blue-500 rounded"></div>
            <span>Start</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500 rounded"></div>
            <span>Goal</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-gray-800 rounded"></div>
            <span>Wall</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-yellow-200 border border-yellow-400 rounded"></div>
            <span>Learned Path</span>
          </div>
        </div>
      </Card>

      {/* Learning Progress */}
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Training Progress: Steps per Episode</h4>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trainingData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="episode" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="steps" 
              stroke="#3b82f6" 
              strokeWidth={2}
              dot={false}
              name="Steps to Goal"
            />
          </LineChart>
        </ResponsiveContainer>
        <p className="mt-4 text-sm text-gray-600">
          Agent learns to reach goal more efficiently over time
        </p>
      </Card>

      {/* Cumulative Reward */}
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Cumulative Reward</h4>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={trainingData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="episode" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="reward" 
              stroke="#10b981" 
              strokeWidth={2}
              dot={false}
              name="Total Reward"
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Q-Learning Concepts */}
      <Card className="p-6 bg-gradient-to-r from-purple-50 to-pink-50">
        <h4 className="text-lg font-semibold mb-3">Key Q-Learning Concepts</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="font-semibold mb-2">Q-Value Update</div>
            <div className="text-sm text-gray-600 font-mono">
              Q(s,a) ← Q(s,a) + α[r + γ·max Q(s&apos;,a&apos;) - Q(s,a)]
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="font-semibold mb-2">Exploration vs Exploitation</div>
            <div className="text-sm text-gray-600">
              ε-greedy: Explore with probability ε, else exploit best action
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="font-semibold mb-2">Learning Rate (α)</div>
            <div className="text-sm text-gray-600">
              Controls how much new information overrides old: 0.1
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="font-semibold mb-2">Discount Factor (γ)</div>
            <div className="text-sm text-gray-600">
              Importance of future rewards: 0.95
            </div>
          </div>
        </div>
      </Card>

      {/* Learned Policy */}
      {episode > 400 && (
        <Card className="p-6 bg-green-50">
          <h4 className="text-lg font-semibold mb-3 text-green-800">✓ Optimal Policy Learned!</h4>
          <p className="text-gray-700">
            The agent has successfully learned an optimal path from start to goal in just {learnedPath.length - 1} steps.
          </p>
          <div className="mt-3 text-sm text-gray-600">
            <p>• Avoids walls automatically</p>
            <p>• Takes shortest path to goal</p>
            <p>• Policy is stored in Q-table</p>
          </div>
        </card>
      )}
    </div>
  );
};

export default QLearningViz;
