import { useState, useRef, useEffect } from 'react';
import { Play, Pause, Car, Users, CheckCircle, AlertCircle } from 'lucide-react';

// Mock JSON data for different videos
const mockVideoStats = {
  video1: {
    "frame_001": {"occupied": 5, "free": 10},
    "frame_002": {"occupied": 6, "free": 9},
    "frame_003": {"occupied": 4, "free": 11},
    "frame_004": {"occupied": 7, "free": 8},
    "frame_005": {"occupied": 5, "free": 10},
    "frame_006": {"occupied": 8, "free": 7},
    "frame_007": {"occupied": 6, "free": 9},
    "frame_008": {"occupied": 9, "free": 6},
    "frame_009": {"occupied": 7, "free": 8},
    "frame_010": {"occupied": 5, "free": 10}
  },
  video2: {
    "frame_001": {"occupied": 12, "free": 8},
    "frame_002": {"occupied": 11, "free": 9},
    "frame_003": {"occupied": 13, "free": 7},
    "frame_004": {"occupied": 10, "free": 10},
    "frame_005": {"occupied": 14, "free": 6},
    "frame_006": {"occupied": 12, "free": 8},
    "frame_007": {"occupied": 9, "free": 11},
    "frame_008": {"occupied": 15, "free": 5},
    "frame_009": {"occupied": 13, "free": 7},
    "frame_010": {"occupied": 11, "free": 9}
  },
  video3: {
    "frame_001": {"occupied": 8, "free": 17},
    "frame_002": {"occupied": 9, "free": 16},
    "frame_003": {"occupied": 7, "free": 18},
    "frame_004": {"occupied": 10, "free": 15},
    "frame_005": {"occupied": 12, "free": 13},
    "frame_006": {"occupied": 8, "free": 17},
    "frame_007": {"occupied": 11, "free": 14},
    "frame_008": {"occupied": 6, "free": 19},
    "frame_009": {"occupied": 9, "free": 16},
    "frame_010": {"occupied": 13, "free": 12}
  }
};

// Header Component
const Header = () => {
  return (
    <header className="bg-white border-b border-gray-200 py-4 px-6 shadow-sm">
      <div className="flex items-center space-x-3">
        <Car className="w-8 h-8 text-blue-600" />
        <h1 className="text-2xl font-bold text-gray-900">Parking Slot Detection Dashboard</h1>
      </div>
    </header>
  );
};

// Left Sidebar Component
const LeftSidebar = ({ selectedVideo, setSelectedVideo, selectedModel, setSelectedModel, isPlaying, onPlayToggle }) => {
  const videoOptions = [
    { value: 'video1', label: 'Parking Lot A (video1.mp4)' },
    { value: 'video2', label: 'Parking Lot B (video2.mp4)' },
    { value: 'video3', label: 'Parking Lot C (video3.mp4)' }
  ];

  const modelOptions = [
    'YOLOv5s',
    'YOLOv5m',
    'YOLOv8n',
    'YOLOv8s',
    'YOLOv8m'
  ];

  return (
    <div className="bg-gray-50 p-6 h-full">
      <div className="space-y-6">
        {/* Video Selection */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-3">
            Select Video File
          </label>
          <select
            value={selectedVideo}
            onChange={(e) => setSelectedVideo(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
          >
            <option value="">Choose a video...</option>
            {videoOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

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
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </div>

        {/* Control Button */}
        <button
          onClick={onPlayToggle}
          disabled={!selectedVideo || !selectedModel}
          className={`w-full py-3 px-4 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center space-x-2 ${
            !selectedVideo || !selectedModel
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : isPlaying
              ? 'bg-red-500 hover:bg-red-600 text-white'
              : 'bg-green-500 hover:bg-green-600 text-white'
          }`}
        >
          {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
          <span>{isPlaying ? 'Stop Detection' : 'Start Detection'}</span>
        </button>

        {/* Current Configuration */}
        <div className="mt-8 p-4 bg-white rounded-lg border border-gray-200">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Configuration</h3>
          <div className="space-y-2 text-sm text-gray-600">
            <div className="flex justify-between">
              <span>Video:</span>
              <span className="font-medium">{selectedVideo || 'None'}</span>
            </div>
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

// Video Display Component
const VideoDisplay = ({ selectedVideo, isPlaying, onTimeUpdate }) => {
  const videoRef = useRef(null);

  useEffect(() => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.play();
      } else {
        videoRef.current.pause();
      }
    }
  }, [isPlaying]);

  const handleTimeUpdate = () => {
    if (videoRef.current && onTimeUpdate) {
      onTimeUpdate(videoRef.current.currentTime);
    }
  };

  return (
    <div className="bg-white p-4 h-full">
      <div className="bg-black rounded-lg h-full flex items-center justify-center relative overflow-hidden">
        {selectedVideo ? (
          <video
            ref={videoRef}
            className="w-full h-full object-contain rounded-lg"
            onTimeUpdate={handleTimeUpdate}
            loop
            muted
          >
            <source src={`${selectedVideo}.mp4`} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        ) : (
          <div className="text-center text-gray-400">
            <Car className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <p className="text-lg">Select a video to begin detection</p>
          </div>
        )}

        {/* Status Indicator */}
        {isPlaying && selectedVideo && (
          <div className="absolute top-4 left-4 bg-red-500 text-white px-3 py-1 rounded-full text-sm font-semibold flex items-center space-x-1">
            <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
            <span>LIVE DETECTION</span>
          </div>
        )}
      </div>
    </div>
  );
};

// Stats Card Component
const StatsCard = ({ title, value, color, icon: Icon, bgColor }) => {
  return (
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
};

// Right Sidebar Component
const RightSidebar = ({ currentStats }) => {
  const total = currentStats.occupied + currentStats.free;
  const occupancyRate = total > 0 ? Math.round((currentStats.occupied / total) * 100) : 0;

  return (
    <div className="bg-gray-50 p-6 h-full">
      <div className="space-y-6">
        {/* Title */}
        <h2 className="text-lg font-semibold text-gray-800 flex items-center space-x-2">
          <Users className="w-5 h-5 text-blue-600" />
          <span>Real-time Statistics</span>
        </h2>

        {/* Stats Cards */}
        <div className="space-y-4">
          <StatsCard
            title="Total Slots"
            value={total}
            color="text-blue-600"
            icon={Car}
            bgColor="bg-blue-50"
          />
          <StatsCard
            title="Occupied"
            value={currentStats.occupied}
            color="text-red-600"
            icon={AlertCircle}
            bgColor="bg-red-50"
          />
          <StatsCard
            title="Available"
            value={currentStats.free}
            color="text-green-600"
            icon={CheckCircle}
            bgColor="bg-green-50"
          />
        </div>

        {/* Occupancy Rate */}
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

        {/* Status Summary */}
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Status Summary</h3>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Utilization:</span>
              <span className={`font-semibold ${
                occupancyRate > 80 ? 'text-red-600' : 
                occupancyRate > 60 ? 'text-yellow-600' : 'text-green-600'
              }`}>
                {occupancyRate > 80 ? 'High' : occupancyRate > 60 ? 'Medium' : 'Low'}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Efficiency:</span>
              <span className="font-semibold text-blue-600">
                {total > 0 ? Math.round((currentStats.free / total) * 100) : 0}% Available
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Footer Status Bar Component
const StatusBar = ({ currentTime, fps, isPlaying }) => {
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <footer className="bg-white border-t border-gray-200 py-3 px-6">
      <div className="flex justify-between items-center text-sm">
        <div className="flex items-center space-x-6">
          <span className="text-gray-600">
            Time: <span className="font-mono font-semibold">{formatTime(currentTime)}</span>
          </span>
          <span className="text-gray-600">
            FPS: <span className="font-semibold">{fps}</span>
          </span>
          <div className={`flex items-center space-x-2 ${isPlaying ? 'text-green-600' : 'text-gray-500'}`}>
            <div className={`w-2 h-2 rounded-full ${isPlaying ? 'bg-green-500' : 'bg-gray-400'}`}></div>
            <span className="font-semibold">{isPlaying ? 'Processing' : 'Paused'}</span>
          </div>
        </div>
        <div className="text-gray-500 text-xs">
          Parking Slot Detection System v1.0
        </div>
      </div>
    </footer>
  );
};

// Main Dashboard Component
const App = () => {
  const [selectedVideo, setSelectedVideo] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [currentStats, setCurrentStats] = useState({ occupied: 0, free: 0 });

  // Get current frame stats based on video time
  const getCurrentFrameStats = (time) => {
    if (!selectedVideo || !mockVideoStats[selectedVideo]) {
      return { occupied: 0, free: 0 };
    }

    // Convert time to frame (assuming 30fps for demo)
    const frameNumber = Math.floor(time * 2) + 1; // Slow down for demo
    const frameKey = `frame_${frameNumber.toString().padStart(3, '0')}`;
    
    const videoStats = mockVideoStats[selectedVideo];
    const frameStats = videoStats[frameKey] || Object.values(videoStats)[frameNumber % Object.keys(videoStats).length];
    
    return frameStats || { occupied: 0, free: 0 };
  };

  const handlePlayToggle = () => {
    setIsPlaying(!isPlaying);
  };

  const handleTimeUpdate = (time) => {
    setCurrentTime(time);
    const frameStats = getCurrentFrameStats(time);
    setCurrentStats(frameStats);
  };

  // Reset stats when video changes
  useEffect(() => {
    if (selectedVideo) {
      setCurrentStats(getCurrentFrameStats(0));
      setCurrentTime(0);
    } else {
      setCurrentStats({ occupied: 0, free: 0 });
      setCurrentTime(0);
    }
    setIsPlaying(false);
  }, [selectedVideo]);

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      <Header />
      
      <div className="flex-1 grid grid-cols-12 gap-0 min-h-0">
        {/* Left Sidebar */}
        <div className="col-span-3 border-r border-gray-200">
          <LeftSidebar
            selectedVideo={selectedVideo}
            setSelectedVideo={setSelectedVideo}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            isPlaying={isPlaying}
            onPlayToggle={handlePlayToggle}
          />
        </div>
        
        {/* Center Video Area */}
        <div className="col-span-6">
          <VideoDisplay
            selectedVideo={selectedVideo}
            isPlaying={isPlaying}
            onTimeUpdate={handleTimeUpdate}
          />
        </div>
        
        {/* Right Sidebar */}
        <div className="col-span-3 border-l border-gray-200">
          <RightSidebar currentStats={currentStats} />
        </div>
      </div>
      
      <StatusBar currentTime={currentTime} fps={30} isPlaying={isPlaying} />
    </div>
  );
};

export default App;