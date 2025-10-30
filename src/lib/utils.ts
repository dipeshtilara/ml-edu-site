import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatCode(code: string): string {
  return code.trim().replace(/\t/g, '    ')
}

export function downloadFile(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

export function getProjectsByTag(projects: any[], tag: string) {
  return projects.filter(project => 
    project.tags.some((t: string) => t.toLowerCase().includes(tag.toLowerCase()))
  )
}

export function getProjectsByDifficulty(projects: any[], difficulty: string) {
  return projects.filter(project => 
    project.difficulty.toLowerCase() === difficulty.toLowerCase()
  )
}

export function searchProjects(projects: any[], query: string) {
  const lowercaseQuery = query.toLowerCase()
  return projects.filter(project =>
    project.title.toLowerCase().includes(lowercaseQuery) ||
    project.description.toLowerCase().includes(lowercaseQuery) ||
    project.tags.some((tag: string) => tag.toLowerCase().includes(lowercaseQuery)) ||
    project.cbse_unit.toLowerCase().includes(lowercaseQuery)
  )
}