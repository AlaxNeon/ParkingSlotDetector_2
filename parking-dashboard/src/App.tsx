import { useState, useEffect } from 'react';
import { Play, Pause, Car, Users, CheckCircle, AlertCircle } from 'lucide-react';

// Utility: Load CSV into JS object
const loadCSV = async (fileName: string) => {
  const response = await fetch(`/data/${fileName}`);
  const text = await response.text();
  const rows = text.trim().split("\n");
  const headers = rows[0].split(",");
  const data: Record<string, { total: number; occupied: number; free: number }> = {};

  for (let i = 1; i < rows.length; i++) {
    const values = rows[i].split(",");
    const entry: Record<string, string> = {};
    headers.forEach((h, j) => {
      entry[h.trim()] = values[j].trim();
    });

    const frameKey = entry["Frame"]; // already like frame_001
    data[frameKey] = {
      total: parseInt(entry["Total Slots"]),
      occupied: parseInt(entry["Occupied"]),
      free: parseInt(entry["Free"]),
    };
  }
  return data;
};

// Header Component
const Header = () => (
  <header className="bg-white border-b border-gray-200 py-4 px-6 shadow-sm">
    <div className="flex items-center space-x-3">
      <Car className="w-8 h-8 text-blue-600" />
      <h1 className="text-2xl font-bold text-gray-900">Parking Slot Detection Dashboard</h1>
    </div>
  </header>
);

// Left Sidebar
const LeftSidebar = ({ selectedModel, setSelectedModel, isPlaying, onPlayToggle }) => {
  const modelOptions = ['YOLOv5s', 'YOLOv8s'];

  return (
    <div className="bg-gray-50 p-6 h-full">
      <div className="space-y-6">
        {/* Model Selection */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-3">
            Detection Model
          </label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
          >
            <option value="">Select model...</option>
            {modelOptions.map((model) => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>

        {/* Play / Pause */}
        <button
          onClick={onPlayToggle}
          disabled={!selectedModel}
          className={`w-full py-3 px-4 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center space-x-2 ${
            !selectedModel
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : isPlaying
              ? 'bg-red-500 hover:bg-red-600 text-white'
              : 'bg-green-500 hover:bg-green-600 text-white'
          }`}
        >
          {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
          <span>{isPlaying ? 'Stop Simulation' : 'Start Simulation'}</span>
        </button>

        {/* Config */}
        <div className="mt-8 p-4 bg-white rounded-lg border border-gray-200">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Configuration</h3>
          <div className="space-y-2 text-sm text-gray-600">
            <div className="flex justify-between">
              <span>Model:</span>
              <span className="font-medium">{selectedModel || 'None'}</span>
            </div>
            <div className="flex justify-between">
              <span>Status:</span>
              <span className={`font-medium ${isPlaying ? 'text-green-600' : 'text-gray-500'}`}>
                {isPlaying ? 'Running' : 'Stopped'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Fake Video Area
const VideoDisplay = ({ currentFrame }) => (
  <div className="bg-white p-4 h-full flex items-center justify-center">
    <div className="bg-black text-white rounded-lg w-full h-full flex flex-col items-center justify-center">
      <Car className="w-16 h-16 mb-4 text-blue-400" />
      <p className="text-lg">Simulating Detection</p>
      <p className="text-sm text-gray-300">Frame: {currentFrame}</p>
    </div>
  </div>
);

// Stats Card
const StatsCard = ({ title, value, color, icon: Icon, bgColor }) => (
  <div className={`${bgColor} p-4 rounded-lg border border-gray-200 shadow-sm`}>
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-gray-600">{title}</p>
        <p className={`text-3xl font-bold ${color}`}>{value}</p>
      </div>
      <Icon className={`w-8 h-8 ${color}`} />
    </div>
  </div>
);

// Right Sidebar
const RightSidebar = ({ currentStats }) => {
  const occupancyRate = currentStats.total > 0
    ? Math.round((currentStats.occupied / currentStats.total) * 100)
    : 0;

  return (
    <div className="bg-gray-50 p-6 h-full">
      <div className="space-y-6">
        <h2 className="text-lg font-semibold text-gray-800 flex items-center space-x-2">
          <Users className="w-5 h-5 text-blue-600" />
          <span>Real-time Statistics</span>
        </h2>

        <div className="space-y-4">
          <StatsCard title="Total Slots" value={currentStats.total} color="text-blue-600" icon={Car} bgColor="bg-blue-50" />
          <StatsCard title="Occupied" value={currentStats.occupied} color="text-red-600" icon={AlertCircle} bgColor="bg-red-50" />
          <StatsCard title="Available" value={currentStats.free} color="text-green-600" icon={CheckCircle} bgColor="bg-green-50" />
        </div>

        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex justify-between items-center mb-3">
            <span className="text-sm font-semibold text-gray-700">Occupancy Rate</span>
            <span className="text-lg font-bold text-gray-800">{occupancyRate}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${occupancyRate}%` }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Dashboard
const App = () => {
  const [selectedModel, setSelectedModel] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [frames, setFrames] = useState<string[]>([]);
  const [modelData, setModelData] = useState<Record<string, any>>({});
  const [currentStats, setCurrentStats] = useState({ total: 0, occupied: 0, free: 0 });

  // load CSV when model changes
  useEffect(() => {
    if (selectedModel) {
      const modelToCSV: Record<string, string> = {
        YOLOv5s: "yolov5s_stats.csv",
        YOLOv8s: "yolov8s_stats.csv"
      };
      loadCSV(modelToCSV[selectedModel]).then((data) => {
        setModelData(data);
        setFrames(Object.keys(data));
        setCurrentFrameIndex(0);
        setCurrentStats({ total: 0, occupied: 0, free: 0 });
      });
    } else {
      setModelData({});
      setFrames([]);
      setCurrentFrameIndex(0);
      setCurrentStats({ total: 0, occupied: 0, free: 0 });
    }
    setIsPlaying(false);
  }, [selectedModel]);

  // play frames automatically
  useEffect(() => {
    if (!isPlaying || frames.length === 0) return;
    const interval = setInterval(() => {
      setCurrentFrameIndex((prev) => (prev + 1) % frames.length);
    }, 1000); // 1 frame per sec
    return () => clearInterval(interval);
  }, [isPlaying, frames]);

  // update stats when frame changes
  useEffect(() => {
    if (frames.length > 0) {
      const frameKey = frames[currentFrameIndex];
      setCurrentStats(modelData[frameKey] || { total: 0, occupied: 0, free: 0 });
    }
  }, [currentFrameIndex, frames, modelData]);

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      <Header />
      <div className="flex-1 grid grid-cols-12 gap-0 min-h-0">
        <div className="col-span-3 border-r border-gray-200">
          <LeftSidebar
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            isPlaying={isPlaying}
            onPlayToggle={() => setIsPlaying(!isPlaying)}
          />
        </div>
        <div className="col-span-6">
          <VideoDisplay currentFrame={frames[currentFrameIndex] || "N/A"} />
        </div>
        <div className="col-span-3 border-l border-gray-200">
          <RightSidebar currentStats={currentStats} />
        </div>
      </div>
    </div>
  );
};

export default App;
