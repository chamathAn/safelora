"use client";

import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";

const CROP_STAGES = ["Initial Crop", "Mid Crop", "Final Crop"] as const;

export default function CropStage() {
  return (
    <Combobox items={CROP_STAGES}>
      <ComboboxInput
        placeholder="Select crop stage"
        readOnly
        className="cursor-pointer"
      />

      <ComboboxContent>
        <ComboboxEmpty>No items found.</ComboboxEmpty>

        <ComboboxList>
          {(item) => (
            <ComboboxItem key={item} value={item}>
              {item}
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
}
