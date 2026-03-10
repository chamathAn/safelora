"use client";
import CropStage from "./crop-stage";
import GpsLocation from "./gps-location";
import ImageUpload from "./image-upload";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { file, z } from "zod";
import { Button } from "./ui/button";
import { io, Socket } from "socket.io-client";
import { useEffect, useRef } from "react";
import { socket } from "@/lib/socket";

const formSchema = z.object({
  cropStage: z.string({ error: "Please select crop stage" }),
  latitude: z.number({
    error: "Please get your location",
  }),
  longitude: z.number({
    error: "Please get your location",
  }),
  image: z.instanceof(File, { message: "Please upload an image" }),
});
type FormValues = z.infer<typeof formSchema>;
export default function Inputs() {
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    const socket = io("http://localhost:5000", {
      transports: ["websocket"],
    });

    socketRef.current = socket;

    socket.on("connect", () => {
      console.log("Connected:", socket.id);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const {
    handleSubmit,
    setValue,
    register,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(formSchema),
  });

  register("image");
  register("cropStage");
  register("latitude");
  register("longitude");

  const onSubmit = async (data: FormValues) => {
    const buffer = await data.image.arrayBuffer();
    socket.emit("start", {
      cropStage: data.cropStage,
      latitude: data.latitude,
      longitude: data.longitude,
      image: buffer,
    });
    console.log("Submitted:", data);
  };
  return (
    <form className="space-y-5" onSubmit={handleSubmit(onSubmit)}>
      <div className="gap-y-4 flex flex-col justify-center items-end">
        <h3 className="text-xl font-semibold font-roboto self-start">
          Upload Image
        </h3>
        <ImageUpload
          onChange={(file) =>
            setValue("image", file as File, { shouldValidate: true })
          }
        />
        {errors.image && <p className="text-red-500">{errors.image.message}</p>}
      </div>
      <div className="gap-y-4 flex justify-between gap-4">
        <h3 className="text-xl font-semibold font-roboto">Crop Growth Stage</h3>
        <div className="flex flex-col gap-2">
          <CropStage
            onChange={(stage) =>
              setValue("cropStage", stage, { shouldValidate: true })
            }
          />
          {errors.cropStage && (
            <p className="text-red-500">{errors.cropStage.message}</p>
          )}
        </div>
      </div>
      <div className="gap-y-4 flex gap-4">
        <h3 className="text-xl font-semibold font-roboto">Location</h3>
        <div className="flex flex-col justify-end items-end gap-2 w-full flex-1">
          <GpsLocation
            onChange={(loc) => {
              setValue("latitude", loc.latitude, { shouldValidate: true });
              setValue("longitude", loc.longitude, { shouldValidate: true });
            }}
          />
          {(errors.latitude || errors.longitude) && (
            <p className="text-red-500">
              {errors.latitude?.message || errors.longitude?.message}
            </p>
          )}
        </div>
      </div>
      <Button type="submit">Analyze Disease</Button>
    </form>
  );
}
