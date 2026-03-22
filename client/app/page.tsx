import Advisory from "@/components/advisory";
import Inputs from "@/components/Inputs";
import LiveUpdates from "@/components/live-updates";

export default function Home() {
  return (
    <main className="container px-4 max-w-xl mx-auto min-h-screen space-y-5">
      <h1 className={`text-4xl font-bold font-poppins`}>Safelora</h1>
      <Inputs />
      <LiveUpdates />
      <Advisory />
    </main>
  );
}
