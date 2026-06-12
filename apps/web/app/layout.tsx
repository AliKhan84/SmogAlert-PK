import type { Metadata } from "next";
import { ClerkProvider } from "@clerk/nextjs";
import localFont from "next/font/local";
import "./globals.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "SmogAlert PK — Pakistan Air Quality Alerts",
  description:
    "Real-time air quality monitoring and 24-hour predictive alerts for Pakistan's 5 major cities.",
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "SmogAlert PK",
  },
  formatDetection: { telephone: false },
  themeColor: "#0d9488",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body className={`${geistSans.variable} antialiased bg-white dark:bg-gray-950`}>
          {children}
        </body>
      </html>
    </ClerkProvider>
  );
}
