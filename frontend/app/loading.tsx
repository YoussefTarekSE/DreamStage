import { AmbientStage } from "@/components/ui/AmbientStage";
import { VinylLoader } from "@/components/ui/VinylLoader";

export default function Loading() {
  return (
    <AmbientStage density="calm">
      <div className="flex min-h-[100dvh] items-center justify-center px-6">
        <div className="glass-panel rounded-lg px-10 py-8">
          <VinylLoader label="Preparing the studio" />
        </div>
      </div>
    </AmbientStage>
  );
}
