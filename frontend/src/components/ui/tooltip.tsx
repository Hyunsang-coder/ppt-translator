"use client";

import * as React from "react";
import { Tooltip as TooltipPrimitive } from "radix-ui";

import { cn } from "@/lib/utils";

const TooltipProvider = TooltipPrimitive.TooltipProvider;
const TooltipRoot = TooltipPrimitive.Tooltip;
const TooltipTrigger = TooltipPrimitive.TooltipTrigger;
const TooltipPortal = TooltipPrimitive.TooltipPortal;
const TooltipContentPrimitive = TooltipPrimitive.TooltipContent;

const TooltipContent = React.forwardRef<
  React.ComponentRef<typeof TooltipContentPrimitive>,
  React.ComponentPropsWithoutRef<typeof TooltipContentPrimitive>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPortal>
    <TooltipContentPrimitive
      ref={ref}
      sideOffset={sideOffset}
      className={cn(
        "z-50 max-w-[240px] rounded-md border border-border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-md",
        "animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95",
        className
      )}
      {...props}
    />
  </TooltipPortal>
));
TooltipContent.displayName = "TooltipContent";

export { TooltipRoot as Tooltip, TooltipTrigger, TooltipContent, TooltipProvider };
