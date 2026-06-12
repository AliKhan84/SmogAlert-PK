import { SignIn } from "@clerk/nextjs";

export default function SignInPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 flex items-center justify-center">
      <SignIn fallbackRedirectUrl="/dashboard" />
    </div>
  );
}
