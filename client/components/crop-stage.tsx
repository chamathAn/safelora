"use client";

import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
type Props = {
  onChange: (value: string | null) => void;
};
const CROP_STAGES = ["Initial Crop", "Mid Crop", "Final Crop"] as const;

export default function CropStage({ onChange }: Props) {
  return (
    <Combobox items={CROP_STAGES} onValueChange={onChange}>
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
