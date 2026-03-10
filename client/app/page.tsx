import Inputs from "@/components/Inputs";
import LiveUpdates from "@/components/live-updates";

export default function Home() {
  return (
    <main className="container px-4 max-w-xl mx-auto min-h-screen">
      <h1 className={`text-4xl font-bold font-poppins`}>Safelora</h1>
      <Inputs />
      <LiveUpdates />
    </main>
  );
}
