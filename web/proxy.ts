import { NextResponse, type NextRequest } from "next/server";

/** A judging deployment (CUE_PASSWORD set) sits behind HTTP Basic auth; local development is open. */
export function proxy(request: NextRequest) {
  const password = process.env.CUE_PASSWORD;
  if (!password) return NextResponse.next();
  const [scheme, encoded] = (request.headers.get("authorization") ?? "").split(" ");
  if (scheme === "Basic" && encoded) {
    const decoded = atob(encoded);
    if (decoded.slice(decoded.indexOf(":") + 1) === password) return NextResponse.next();
  }
  return new NextResponse("Cue the Boom: the password is in the submission.", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="Cue the Boom"' },
  });
}

export const config = { matcher: "/((?!_next/static|_next/image|favicon.ico).*)" };
