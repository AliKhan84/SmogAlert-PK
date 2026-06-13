"use client";

import Link from "next/link";
import { SignInButton, SignUpButton, SignOutButton, UserButton, useUser } from "@clerk/nextjs";

export function NavAuth() {
  const { isSignedIn, isLoaded } = useUser();

  if (!isLoaded) return null;

  if (isSignedIn) {
    return (
      <div className="flex items-center gap-3">
        <Link
          href="/dashboard"
          className="text-sm text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
        >
          Dashboard
        </Link>
        <UserButton />
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <SignInButton mode="redirect">
        <button className="text-sm text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white">
          Sign in
        </button>
      </SignInButton>
      <SignUpButton mode="redirect">
        <button className="bg-teal-600 text-white text-sm px-4 py-1.5 rounded-lg hover:bg-teal-700 transition-colors">
          Get alerts free
        </button>
      </SignUpButton>
    </div>
  );
}
