import { useState, useEffect, useRef } from "react";
import { Car, Users, CheckCircle, AlertCircle } from "lucide-react";

// Utility: Load CSV into JS object
const loadCSV = async (fileName: string) => {
  const response = await fetch(`/model/${fileName}`);
  const text = await response.text();
  const rows = text.trim().split("\n");
  const headers = rows[0].split(",");

  type FrameData = {
    frame: number;
    total: number;
    occupied: number;
    free: number;
    rate: number;
  };

  const data: Record<number, FrameData> = {};

  for (let i = 1; i < rows.length; i++) {
    const values = rows[i].split(",");
    const entry: Record<string, string> = {};
    headers.forEach((h, j) => {
      entry[h.trim()] = values[j]?.trim();
    });

    const frameNum = parseInt(entry["frame"]);
    data[frameNum] = {
      frame: frameNum,
      total: parseInt(entry["total_spaces"]),
      occupied: parseInt(entry["occupied"]),
      free: parseInt(entry["free"]),
      rate: parseFloat(entry["occupancy_rate"]),
    };
  }
  return data;
};

// Header Component
const Header = () => (
  <header className="bg-white border-b border-gray-200 py-4 px-6 shadow-sm">
    <div className="flex items-center space-x-3">
      <Car className="w-8 h-8 text-blue-600" />
      <h1 className="text-2xl font-bold text-gray-900">
        Parking Slot Detection Dashboard
      </h1>
    </div>
  </header>
);

// Left Sidebar
const LeftSidebar = ({ isPlaying, onPlayToggle }) => (
  <div className="bg-gray-50 p-6 h-full">
    <div className="space-y-6">
      <button
        onClick={onPlayToggle}
        className={`w-full py-3 px-4 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center space-x-2 ${
          isPlaying
            ? "bg-red-500 hover:bg-red-600 text-white"
            : "bg-green-500 hover:bg-green-600 text-white"
        }`}
      >
        <span>{isPlaying ? "Stop Video" : "Start Video"}</span>
      </button>
    </div>
  </div>
);

// Video Display
const VideoDisplay = ({ videoRef }) => (
  <div className="bg-white p-4 h-full flex items-center justify-center">
    <video
      ref={videoRef}
      src="/model/enhanced_yolo_parking.mp4"
      controls
      muted
      className="rounded-lg w-full h-full"
    />
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
  const occupancyRate = currentStats.rate
    ? Math.round(currentStats.rate)
    : 0;

  return (
    <div className="bg-gray-50 p-6 h-full">
      <div className="space-y-6">
        <h2 className="text-lg font-semibold text-gray-800 flex items-center space-x-2">
          <Users className="w-5 h-5 text-blue-600" />
          <span>Real-time Statistics</span>
        </h2>

        <div className="space-y-4">
          <StatsCard
            title="Total Slots"
            value={currentStats.total}
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

        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex justify-between items-center mb-3">
            <span className="text-sm font-semibold text-gray-700">
              Occupancy Rate
            </span>
            <span className="text-lg font-bold text-gray-800">
              {occupancyRate}%
            </span>
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

// Main App
const App = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [modelData, setModelData] = useState<Record<number, any>>({});
  const [currentStats, setCurrentStats] = useState({
    total: 0,
    occupied: 0,
    free: 0,
    rate: 0,
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const fps = 30; // set video FPS (your CSV is at 30fps)

  // Load CSV once
  useEffect(() => {
    loadCSV("parking_analysis.csv").then((data) => {
      setModelData(data);
    });
  }, []);

  // Sync stats with video time
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      const frameIndex = Math.floor(video.currentTime * fps) + 1;
        if (modelData[frameIndex]) {
          setCurrentStats(modelData[frameIndex]);
        }
    };

    video.addEventListener("timeupdate", handleTimeUpdate);
    return () => {
      video.removeEventListener("timeupdate", handleTimeUpdate);
    };
  }, [modelData]);

  // Play/pause toggle
  const handlePlayToggle = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
    setIsPlaying(!isPlaying);
  };

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      <Header />
      <div className="flex-1 grid grid-cols-12 gap-0 min-h-0">
        <div className="col-span-3 border-r border-gray-200">
          <LeftSidebar isPlaying={isPlaying} onPlayToggle={handlePlayToggle} />
        </div>
        <div className="col-span-6">
          <VideoDisplay videoRef={videoRef} />
        </div>
        <div className="col-span-3 border-l border-gray-200">
          <RightSidebar currentStats={currentStats} />
        </div>
      </div>
    </div>
  );
};

export default App;
