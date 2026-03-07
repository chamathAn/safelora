"use client";

import { MapPin } from "lucide-react";
import { useState } from "react";
import { Button } from "./ui/button";
import { Skeleton } from "./ui/skeleton";

type Weather = {
  temperature: number;
  windspeed: number;
  time: string;
};

type Props = {
  onChange: (loc: { latitude: number; longitude: number }) => void;
};
export default function GpsLocation({ onChange }: Props) {
  const [location, setLocation] = useState<{
    latitude: number;
    longitude: number;
  } | null>(null);

  //   const [weather, setWeather] = useState<Weather | null>(null);
  const [loading, setLoading] = useState(false);

  const getLocation = () => {
    if (!navigator.geolocation) {
      alert("Geolocation is not supported by this browser.");
      return;
    }

    setLoading(true);

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        onChange({ latitude, longitude });
        setLocation({ latitude, longitude });

        // try {
        //   const res = await fetch(
        //     `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current_weather=true`,
        //   );

        //   const data = await res.json();

        //   const cw = data.current_weather;

        //   setWeather({
        //     temperature: cw.temperature,
        //     windspeed: cw.windspeed,
        //     time: cw.time,
        //   });
        // } catch (err) {
        //   console.error(err);
        //   alert("Failed to retrieve weather data");
        // }

        setLoading(false);
      },
      (error) => {
        console.error(error);
        alert("Unable to retrieve location.");
        setLoading(false);
      },
    );
  };

  return (
    <div className="flex flex-col items-end gap-4">
      <Button onClick={getLocation} variant="outline" size="icon">
        <MapPin />
      </Button>

      {loading && (
        <div className="space-y-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
        </div>
      )}

      {location && (
        <div className="font-roboto text-end">
          <p>Latitude: {location.latitude}</p>
          <p>Longitude: {location.longitude}</p>
        </div>
      )}

      {/* {weather && (
        <div>
          <p>Temperature: {weather.temperature} °C</p>
          <p>Wind Speed: {weather.windspeed} km/h</p>
          <p>Time: {weather.time}</p>
        </div>
      )} */}
    </div>
  );
}
