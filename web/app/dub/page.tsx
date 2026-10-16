import Link from "next/link";
import Studio from "@/components/Studio";

/** The old Dub Mode address: the same studio. */
export default function DubPage() {
  return (
    <main>
      <p className="muted"><Link href="/">Cue the Boom</Link> · perform over a clip</p>
      <Studio />
    </main>
  );
}
