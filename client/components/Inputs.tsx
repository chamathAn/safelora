import CropStage from "./crop-stage";
import GpsLocation from "./gps-location";
import ImageUpload from "./image-upload";

export default function Inputs() {
  return (
    <div className="space-y-5">
      <div className="gap-y-4 flex flex-col justify-center items-end">
        <h3 className="text-xl font-semibold font-roboto self-start">
          Upload Image
        </h3>
        <ImageUpload />
      </div>
      <div className="gap-y-4 flex justify-between gap-4">
        <h3 className="text-xl font-semibold font-roboto">Crop Growth Stage</h3>
        <CropStage />
      </div>
      <div className="gap-y-4 flex justify-between gap-4">
        <h3 className="text-xl font-semibold font-roboto">Location</h3>
        <GpsLocation />
      </div>
    </div>
  );
}
