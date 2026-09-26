import { useState, useEffect, useRef } from 'react';
import {
  UserCheck, TrendingDown, MinusCircle, Bookmark, RefreshCw, ShieldCheck,
  Check, Loader2, ChevronRight, AlertTriangle,
} from 'lucide-react';
import { clients, unlearningLogs } from '../data/mockData';

const steps = [
  { id: 1, label: 'Select Target Client', icon: UserCheck, desc: 'Choose the client whose data contribution should be removed.' },
  { id: 2, label: 'Estimate Influence', icon: TrendingDown, desc: 'Calculate gradient-based influence of the selected client on the global model.' },
  { id: 3, label: 'Remove Influence', icon: MinusCircle, desc: 'Subtract the estimated contribution from the global model weights.' },
  { id: 4, label: 'Checkpoint / Rollback', icon: Bookmark, desc: 'Store an intermediate checkpoint or roll back to a pre-contribution state.' },
  { id: 5, label: 'Repair Training', icon: RefreshCw, desc: 'Run adaptive repair rounds with remaining clients to restore model utility.' },
  { id: 6, label: 'Verify Forgetting', icon: ShieldCheck, desc: 'Run membership inference and model distance checks to verify unlearning success.' },
];

const stepDurations = [800, 1800, 2400, 1200, 4000, 2000];

interface UnlearningProps {
  initialClientId?: string;
}

export default function Unlearning({ initialClientId }: UnlearningProps) {
  const [selectedClient, setSelectedClient] = useState(initialClientId ?? '');
  const [running, setRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [showConfirm, setShowConfirm] = useState(false);
  const [done, setDone] = useState(false);
  const [logs, setLogs] = useState<typeof unlearningLogs>([]);
  const [logIndex, setLogIndex] = useState(0);
  const logsRef = useRef<HTMLDivElement>(null);

  const client = clients.find((c) => c.id === selectedClient);

  const totalProgress = done ? 100 : running ? Math.round(((completedSteps.length) / steps.length) * 100) : 0;

  useEffect(() => {
    if (!running) return;
    if (currentStep > steps.length) {
      setDone(true);
      setRunning(false);
      return;
    }
    const stepIdx = currentStep - 1;
    if (stepIdx < 0) return;
    const timer = setTimeout(() => {
      setCompletedSteps((prev) => [...prev, currentStep]);
      setCurrentStep((s) => s + 1);
    }, stepDurations[stepIdx] ?? 1500);
    return () => clearTimeout(timer);
  }, [running, currentStep]);

  // Log streaming effect
  useEffect(() => {
    if (!running && !done) return;
    if (logIndex >= unlearningLogs.length) return;
    const timer = setTimeout(() => {
      setLogs((prev) => [...prev, unlearningLogs[logIndex]]);
      setLogIndex((i) => i + 1);
      if (logsRef.current) {
        logsRef.current.scrollTop = logsRef.current.scrollHeight;
      }
    }, 600);
    return () => clearTimeout(timer);
  }, [running, done, logIndex]);

  function startUnlearning() {
    setShowConfirm(false);
    setRunning(true);
    setCurrentStep(1);
    setCompletedSteps([]);
    setDone(false);
    setLogs([]);
    setLogIndex(0);
  }

  function reset() {
    setRunning(false);
    setCurrentStep(0);
    setCompletedSteps([]);
    setDone(false);
    setLogs([]);
    setLogIndex(0);
    setSelectedClient('');
  }

  const logColors: Record<string, string> = {
    info: '#6a8caa',
    warn: '#f59e0b',
    success: '#22c55e',
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Step pipeline */}
      <div
        className="bg-card rounded-2xl border p-6"
        style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="font-display font-semibold text-base" style={{ color: '#1a3352' }}>Unlearning Pipeline</h2>
            <p className="text-xs mt-0.5" style={{ color: '#6a8caa' }}>Step-by-step federated unlearning procedure</p>
          </div>
          {done && (
            <span className="px-3 py-1.5 rounded-full text-xs font-semibold" style={{ background: 'rgba(34,197,94,0.12)', color: '#22c55e' }}>
              ✓ Unlearning Complete
            </span>
          )}
        </div>

        {/* Steps */}
        <div className="flex flex-col lg:flex-row items-start lg:items-center gap-2 lg:gap-0">
          {steps.map((step, idx) => {
            const isComplete = completedSteps.includes(step.id);
            const isActive = currentStep === step.id;
            const Icon = step.icon;
            return (
              <div key={step.id} className="flex lg:flex-col items-center lg:flex-1 gap-2 lg:gap-0">
                <div className="flex lg:flex-col items-center lg:items-center gap-2 lg:gap-2 flex-1 lg:w-full">
                  <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0 transition-all"
                    style={{
                      background: isComplete
                        ? '#2d6a9f'
                        : isActive
                        ? 'rgba(45,106,159,0.15)'
                        : '#eef3f9',
                      border: isActive ? '2px solid #2d6a9f' : '2px solid transparent',
                    }}
                  >
                    {isComplete ? (
                      <Check size={16} color="#fff" />
                    ) : isActive ? (
                      <Loader2 size={16} color="#2d6a9f" className="animate-spin-slow" />
                    ) : (
                      <Icon size={16} color={isComplete ? '#fff' : '#a0bdd0'} />
                    )}
                  </div>
                  <div className="lg:text-center lg:mt-2 flex-1 lg:flex-none">
                    <div
                      className="text-xs font-semibold"
                      style={{ color: isComplete ? '#2d6a9f' : isActive ? '#1a3352' : '#a0bdd0' }}
                    >
                      Step {step.id}
                    </div>
                    <div
                      className="text-xs mt-0.5 leading-tight max-w-[100px]"
                      style={{ color: isComplete ? '#1a3352' : isActive ? '#6a8caa' : '#c5d5e8' }}
                    >
                      {step.label}
                    </div>
                  </div>
                </div>
                {idx < steps.length - 1 && (
                  <ChevronRight size={14} color={completedSteps.includes(step.id + 1) || isComplete ? '#7ab3d8' : '#d8e6f2'} className="hidden lg:block" />
                )}
              </div>
            );
          })}
        </div>

        {/* Progress bar */}
        {(running || done) && (
          <div className="mt-6">
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span style={{ color: '#6a8caa' }}>Overall Progress</span>
              <span className="font-mono font-semibold" style={{ color: '#2d6a9f' }}>{totalProgress}%</span>
            </div>
            <div className="h-2 rounded-full" style={{ background: '#eef3f9' }}>
              <div
                className="h-2 rounded-full transition-all duration-500"
                style={{ width: `${totalProgress}%`, background: 'linear-gradient(90deg, #2d6a9f, #7ab3d8)' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Config + logs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Config */}
        <div
          className="bg-card rounded-2xl border p-5 space-y-4"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)' }}
        >
          <h3 className="font-display font-semibold text-sm" style={{ color: '#1a3352' }}>Unlearning Configuration</h3>

          {/* Client select */}
          <div>
            <label className="text-xs font-semibold block mb-1.5" style={{ color: '#6a8caa' }}>Target Client</label>
            <select
              value={selectedClient}
              onChange={(e) => setSelectedClient(e.target.value)}
              disabled={running}
              className="w-full rounded-xl border px-3 py-2.5 text-sm font-medium appearance-none cursor-pointer"
              style={{
                borderColor: 'var(--color-border)',
                background: running ? '#f5f3ee' : '#fff',
                color: '#1a3352',
                outline: 'none',
              }}
            >
              <option value="">— Select a client —</option>
              {clients.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.id} — {c.name} (influence: {(c.influenceScore * 100).toFixed(1)}%)
                </option>
              ))}
            </select>
          </div>

          {/* Config display */}
          {client && (
            <div className="space-y-2.5">
              {[
                { label: 'Estimated Influence', value: `${(client.influenceScore * 100).toFixed(1)}%` },
                { label: 'Contribution Score', value: `${(client.contributionScore * 100).toFixed(1)}%` },
                { label: 'Selected Strategy', value: client.influenceScore > 0.3 ? 'Deep Unlearning' : 'Lightweight Unlearning' },
                { label: 'Compression Level', value: client.compressionLevel },
                { label: 'Repair Rounds', value: client.influenceScore > 0.3 ? '3' : '1' },
                { label: 'Checkpoint Freq.', value: client.compute > 70 ? 'Every 1 step' : 'Every 3 steps' },
              ].map((row) => (
                <div key={row.label} className="flex justify-between items-center py-1.5 border-b" style={{ borderColor: '#eef3f9' }}>
                  <span className="text-xs" style={{ color: '#6a8caa' }}>{row.label}</span>
                  <span className="text-xs font-semibold font-mono" style={{ color: '#1a3352' }}>{row.value}</span>
                </div>
              ))}
            </div>
          )}

          {/* Action buttons */}
          <div className="flex gap-2 pt-1">
            {!running && !done && (
              <button
                disabled={!selectedClient}
                onClick={() => setShowConfirm(true)}
                className="flex-1 py-2.5 rounded-xl text-sm font-semibold text-white cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                style={{ background: '#2d6a9f' }}
              >
                Start Unlearning
              </button>
            )}
            {(running || done) && (
              <button
                onClick={reset}
                className="flex-1 py-2.5 rounded-xl text-sm font-semibold cursor-pointer border"
                style={{ borderColor: 'var(--color-border)', color: '#6a8caa', background: '#fff' }}
              >
                Reset
              </button>
            )}
          </div>
        </div>

        {/* Logs */}
        <div
          className="bg-card rounded-2xl border p-5 flex flex-col"
          style={{ borderColor: 'var(--color-border)', boxShadow: '0 1px 4px rgba(26,51,82,0.06)', minHeight: 320 }}
        >
          <h3 className="font-display font-semibold text-sm mb-3" style={{ color: '#1a3352' }}>Processing Log</h3>
          <div
            ref={logsRef}
            className="flex-1 overflow-y-auto space-y-1 rounded-xl p-3"
            style={{ background: '#0f1f33', fontFamily: 'var(--font-mono)', fontSize: 11, minHeight: 240, maxHeight: 280 }}
          >
            {logs.length === 0 && (
              <div className="text-center py-8" style={{ color: 'rgba(122,179,216,0.4)' }}>
                Awaiting process start...
              </div>
            )}
            {logs.map((log, i) => (
              <div key={i} className="flex gap-2 items-start leading-relaxed">
                <span style={{ color: 'rgba(122,179,216,0.5)', whiteSpace: 'nowrap' }}>{log.time}</span>
                <span
                  className="uppercase shrink-0 text-[9px] font-bold px-1 py-0.5 rounded"
                  style={{
                    background: logColors[log.level] + '25',
                    color: logColors[log.level],
                  }}
                >
                  {log.level}
                </span>
                <span style={{ color: log.level === 'success' ? '#22c55e' : log.level === 'warn' ? '#f59e0b' : 'rgba(220,235,248,0.85)' }}>
                  {log.msg}
                </span>
              </div>
            ))}
            {running && logIndex < unlearningLogs.length && (
              <div className="flex gap-1 items-center pt-1" style={{ color: '#7ab3d8' }}>
                <span className="animate-pulse-soft">▋</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Confirmation modal */}
      {showConfirm && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{ background: 'rgba(26,51,82,0.4)', backdropFilter: 'blur(4px)' }}
        >
          <div
            className="w-full max-w-sm rounded-2xl p-6 animate-fade-in"
            style={{ background: '#fff', boxShadow: '0 8px 40px rgba(26,51,82,0.2)' }}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: 'rgba(224,92,92,0.1)' }}>
                <AlertTriangle size={18} color="#e05c5c" />
              </div>
              <div>
                <div className="font-display font-semibold text-base" style={{ color: '#1a3352' }}>Confirm Unlearning</div>
                <div className="text-xs mt-0.5" style={{ color: '#6a8caa' }}>This action modifies the global model.</div>
              </div>
            </div>
            <p className="text-sm mb-5" style={{ color: '#6a8caa' }}>
              You are about to remove the contribution of <strong style={{ color: '#1a3352' }}>{selectedClient}</strong> from the global model.
              This will run the full unlearning and repair pipeline. This action cannot be undone without re-training.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setShowConfirm(false)}
                className="flex-1 py-2.5 rounded-xl text-sm font-semibold border cursor-pointer"
                style={{ borderColor: 'var(--color-border)', color: '#6a8caa', background: '#fff' }}
              >
                Cancel
              </button>
              <button
                onClick={startUnlearning}
                className="flex-1 py-2.5 rounded-xl text-sm font-semibold text-white cursor-pointer"
                style={{ background: '#2d6a9f' }}
              >
                Confirm & Start
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
