"use client";

import { Upload, X } from "lucide-react";
import * as React from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  FileUpload,
  FileUploadDropzone,
  FileUploadItem,
  FileUploadItemDelete,
  FileUploadItemMetadata,
  FileUploadItemPreview,
  FileUploadList,
  FileUploadTrigger,
} from "@/components/ui/file-upload";

type Props = {
  onChange: (file: File | null) => void;
};

export default function ImageUpload({ onChange }: Props) {
  const [file, setFile] = React.useState<File[]>([]);

  const onFileReject = React.useCallback((file: File, message: string) => {
    toast(message, {
      description: `"${file.name.length > 20 ? `${file.name.slice(0, 20)}...` : file.name}" has been rejected`,
    });
  }, []);

  return (
    <FileUpload
      maxFiles={1}
      maxSize={5 * 1024 * 1024}
      accept="image/png,image/jpeg"
      className="w-full max-w-md"
      value={file}
      onValueChange={(f) => {
        setFile(f);
        onChange(f[0] ?? null);
      }}
      onFileReject={onFileReject}
      multiple={false}
      disabled={file.length >= 1}
    >
      <FileUploadDropzone>
        <div className="flex flex-col items-center gap-1 text-center font-roboto">
          <div className="flex items-center justify-center rounded-full border p-2.5">
            <Upload className="size-6 text-muted-foreground" />
          </div>

          <p className="font-medium text-sm">Drag & drop image here</p>

          <p className="text-muted-foreground text-xs">
            Or click to browse (max 1 file, up to 5MB)
          </p>
        </div>

        <FileUploadTrigger asChild>
          <Button variant="outline" size="sm" className="mt-2 w-fit">
            Browse images
          </Button>
        </FileUploadTrigger>
      </FileUploadDropzone>

      <FileUploadList>
        {file.map((file, index) => (
          <FileUploadItem key={index} value={file}>
            <FileUploadItemPreview />
            <FileUploadItemMetadata />

            <FileUploadItemDelete asChild>
              <Button variant="ghost" size="icon" className="size-7">
                <X />
              </Button>
            </FileUploadItemDelete>
          </FileUploadItem>
        ))}
      </FileUploadList>
    </FileUpload>
  );
}
