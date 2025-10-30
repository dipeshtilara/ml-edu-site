'use client'

import React from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { BookOpen, Brain, Home, Info, MapPin } from 'lucide-react'

export function Navbar() {
  const pathname = usePathname()

  const navigation = [
    { name: 'Home', href: '/', icon: Home },
    { name: 'Projects', href: '/projects', icon: Brain },
    { name: 'CBSE Mapping', href: '/cbse-mapping', icon: MapPin },
    { name: 'About', href: '/about', icon: Info },
  ]

  return (
    <nav className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center space-x-2">
            <Brain className="h-8 w-8 text-primary" />
            <Link href="/" className="text-xl font-bold text-primary">
              AI Learning Hub
            </Link>
          </div>
          
          <div className="hidden md:flex items-center space-x-1">
            {navigation.map((item) => {
              const Icon = item.icon
              return (
                <Link key={item.name} href={item.href}>
                  <Button
                    variant={pathname === item.href ? "default" : "ghost"}
                    className="flex items-center space-x-2"
                  >
                    <Icon className="h-4 w-4" />
                    <span>{item.name}</span>
                  </Button>
                </Link>
              )
            })}
          </div>

          <div className="md:hidden">
            <Button variant="ghost" size="icon">
              <BookOpen className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </div>
    </nav>
  )
}