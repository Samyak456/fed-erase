import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip, Legend, BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts';
import { comparisonMethods } from '../data/mockData';
import { Info } from 'lucide-react';

const costColors: Record<string, string> = {
  'Negligible': '#22c55e',
  'Low': '#5b9bd5',
  'Medium': '#f59e0b',
  'High': '#e05c5c',
  'Very High': '#9b2335',
};

const methodColors = ['#2d6a9f', '#7ab3d8', '#f59e0b', '#9ecbe8'];

const radarData = [
  { metric: 'Forgetting', FedErase: 94, 'Full Retrain': 99, 'Naive Sub.': 38, Checkpoint: 76 },
  { metric: 'Utility', FedErase: 85, 'Full Retrain': 84, 'Naive Sub.': 71, Checkpoint: 81 },
  { metric: 'Speed', FedErase: 90, 'Full Retrain': 20, 'Naive Sub.': 99, Checkpoint: 65 },
  { metric: 'Comm. Eff.', FedErase: 88, 'Full Retrain': 10, 'Naive Sub.': 99, Checkpoint: 55 },
  { metric: 'Scalability', FedErase: 85, 'Full Retrain': 30, 'Naive Sub.': 90, Checkpoint: 60 },
];

export default function ModelComparison() {
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Disclaimer */}
      <div
        className="rounded-xl px-4 py-3 border flex items-start gap-2.5 text-xs"
        style={{ background: 'rgba(122,179,216,0.08)', borderColor: '#d8e6f2', color: '#6a8caa' }}
      >
        <Info size={14} className="shrink-0 mt-0.5" />
        <span>
          This comparison presents metrics across four unlearning approaches using the same federated model and target client.
          Results reflect trade-offs inherent to each method — no single method is universally optimal across all deployment contexts.
        </span>
      </div>

      {/* Comparison table */}
      <div
        className="bg-card rounded-2xl border overflow-hidden"
        style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
      >
        <div className="px-6 py-4 border-b" style={{ borderColor: 'var(--color-border)' }}>
          <h2 className="font-display font-semibold text-base" style={{ color: '#1a3352' }}>Method Comparison</h2>
          <p className="text-xs mt-0.5" style={{ color: '#6a8caa' }}>Evaluated on the same global model with Client C-04 as the unlearning target.</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead style={{ background: '#f8fbfe' }}>
              <tr>
                {['Method', 'Unlearning Cost', 'Training Time', 'Communication', 'Model Utility', 'Forgetting Score'].map((h) => (
                  <th key={h} className="px-5 py-3 text-left text-xs font-semibold uppercase tracking-wide" style={{ color: '#6a8caa' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {comparisonMethods.map((m, idx) => (
                <tr
                  key={m.name}
                  className="border-t"
                  style={{
                    borderColor: '#eef3f9',
                    background: m.highlight ? 'rgba(45,106,159,0.04)' : idx % 2 === 0 ? '#fff' : '#fafcfe',
                  }}
                >
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ background: methodColors[idx] }}
                      />
                      <span className="font-semibold text-sm" style={{ color: '#1a3352' }}>{m.name}</span>
                      {m.highlight && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-bold" style={{ background: '#ddeaf7', color: '#2d6a9f' }}>
                          This work
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-5 py-4">
                    <span
                      className="px-2 py-0.5 rounded-full text-xs font-medium"
                      style={{ background: (costColors[m.unlearningCost] ?? '#6a8caa') + '18', color: costColors[m.unlearningCost] ?? '#6a8caa' }}
                    >
                      {m.unlearningCost}
                    </span>
                  </td>
                  <td className="px-5 py-4 font-mono text-xs" style={{ color: '#1a3352' }}>{m.trainingTime}</td>
                  <td className="px-5 py-4">
                    <span
                      className="px-2 py-0.5 rounded-full text-xs font-medium"
                      style={{ background: (costColors[m.communication] ?? '#6a8caa') + '18', color: costColors[m.communication] ?? '#6a8caa' }}
                    >
                      {m.communication}
                    </span>
                  </td>
                  <td className="px-5 py-4 font-mono text-sm font-semibold" style={{ color: '#1a3352' }}>{m.modelUtility}</td>
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-1.5 rounded-full" style={{ background: '#eef3f9', maxWidth: 80 }}>
                        <div
                          className="h-1.5 rounded-full"
                          style={{ width: `${m.forgettingScore}%`, background: m.highlight ? '#2d6a9f' : '#9ecbe8' }}
                        />
                      </div>
                      <span className="font-mono text-xs font-semibold" style={{ color: '#1a3352' }}>{m.forgettingEffectiveness}</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Radar */}
        <div
          className="bg-card rounded-2xl border p-5"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm mb-4" style={{ color: '#1a3352' }}>Multi-Dimensional Comparison</h3>
          <ResponsiveContainer width="100%" height={260}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#eef3f9" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: '#6a8caa' }} />
              <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 9, fill: '#c5d5e8' }} />
              {['FedErase', 'Full Retrain', 'Naive Sub.', 'Checkpoint'].map((name, i) => (
                <Radar key={name} dataKey={name} stroke={methodColors[i]} fill={methodColors[i]} fillOpacity={0.08} strokeWidth={i === 0 ? 2.5 : 1.5} />
              ))}
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 11 }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Forgetting vs utility scatter-style bar */}
        <div
          className="bg-card rounded-2xl border p-5"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm mb-4" style={{ color: '#1a3352' }}>Utility vs Forgetting Score</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart
              data={comparisonMethods.map((m, i) => ({ name: m.name, utility: m.utilityScore, forgetting: m.forgettingScore, fill: methodColors[i] }))}
              margin={{ top: 4, right: 8, bottom: 30, left: -20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#eef3f9" />
              <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#6a8caa' }} angle={-20} textAnchor="end" />
              <YAxis domain={[30, 102]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 10, fill: '#6a8caa' }} />
              <Tooltip formatter={(v) => [`${v}%`]} contentStyle={{ borderRadius: 8, border: '1px solid #d8e6f2', fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Bar dataKey="utility" fill="#2d6a9f" radius={[4, 4, 0, 0]} name="Model Utility" />
              <Bar dataKey="forgetting" fill="#9ecbe8" radius={[4, 4, 0, 0]} name="Forgetting Score" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
